## 1. Summary

This RFC proposes a comprehensive refactoring of the output processing pipeline in vllm-omni. The current `MultimodalOutputProcessor` and `OmniRequestState` classes have grown organically to handle multiple output modalities (text, image, audio, latents) but lack a clear, extensible architecture. A significant portion of the current code is **dead or unreachable**. We propose first simplifying the existing processor by removing dead code, then introducing a **Modality Registry**, a **Router abstraction**, and **composable output handlers** to support emerging multi-output models (e.g., DeepSeek with text+video+image).

## 2. Motivation

### Current Architecture

The output processing pipeline is concentrated in `vllm_omni/engine/output_processor.py` (469 lines) with two main classes:

- **`MultimodalOutputProcessor`** (extends `vllm.v1.engine.output_processor.OutputProcessor`): Routes `EngineCoreOutput` objects by `output_type` string to modality-specific handler methods.
- **`OmniRequestState`** (extends `RequestState`): Accumulates multimodal tensors per-request during generation, with deferred concatenation.

### Problems

#### P0: Significant Dead / Unreachable Code

A thorough audit of the output processor reveals that **a large portion of the code is never executed** in any real production path. This dead code adds confusion, increases maintenance burden, and obscures the actual data flow. See Section 3 for the full dead code inventory.

#### P1: Rigid String-Based Routing

The `_route_and_normalize` method uses hardcoded string matching:

```python
if output_type == "image":
    self._process_image_output(eco)
elif output_type in ("text+image", "text,image", "image+text"):
    self._process_text_image_output(eco)
elif output_type in ("latents", "latent"):
    self._process_latents_output(eco)
elif output_type in ("audio", "speech"):
    self._process_audio_output(eco)
elif output_type == "text":
    self._process_text_output(eco)
```

Adding a new modality (e.g., `video`, `text+video`, `text+video+image`) requires modifying this central dispatch chain.

#### P2: No Composable Multi-Output Support

Models like DeepSeek may produce **text + video + image** simultaneously. The current architecture handles at most two modalities (e.g., `text+image`) via a single combined handler. There is no composition mechanism for arbitrary output type combinations.

#### P3: Inconsistent Key Extraction

Each `_process_*_output` method has its own hardcoded key lists:

- Image: `("image", "images", "pixel_values", "pixels")`
- Audio: `("audio", "audios", "wav", "waveform", "audio_pcm", "pcm")`
- Latents: `("latent", "latents", "z", "posterior")`

There is no standard contract between model output format and the output processor.

#### P4: Monolithic Accumulation Logic

`OmniRequestState.add_multimodal_tensor` handles all modalities in a single method with complex branching for dict/tensor/list inputs and nested dict merging (~65 lines). The special-case for audio tensor concatenation in `_consolidate_multimodal_tensors` (skipping `torch.cat` due to shape inconsistency) further indicates the need for per-modality accumulation strategies.

#### P5: Missing Modalities

- **Video output**: No dedicated `"video"` routing branch exists. Video generation would currently fall through to the heuristic fallback.
- **Text+Video, Text+Video+Image**: No handling exists for these combinations, which future models (e.g., DeepSeek multimodal) will require.

## 3. Dead Code Audit and Simplification Plan

A line-by-line analysis of `output_processor.py` cross-referenced with all callers and stage configs reveals the following dead/unused code:

### 3.1 Dead Code Inventory

#### D1: `register_handler()` — Never Called (lines 281-292)

The `register_handler(modality, handler)` method is defined on `MultimodalOutputProcessor` and initializes `self.output_handlers: dict[str, Callable]`. However, **no code in the entire codebase calls `register_handler()`**. The `output_handlers` dict is always empty, making the custom handler dispatch path in `_route_and_normalize` (lines 373-379) unreachable.

**Evidence**: `rg "register_handler" vllm_omni/` only finds the definition and the docstring reference. Zero callers.

**Action**: Remove `register_handler()`, `self.output_handlers`, and the custom handler dispatch block.

#### D2: `_reqid_to_mm_type` — Written but Never Read (line 278, 346-350)

The `_reqid_to_mm_type: dict[str, str]` is populated in `process_outputs()`:

```python
self._reqid_to_mm_type.clear()
for eco in engine_core_outputs:
    mm_type = (self.engine_core_output_type or "").lower()
    if mm_type:
        self._reqid_to_mm_type[eco.request_id] = mm_type
```

But this dict is **never read anywhere** — not in this class, not in any subclass, not in any caller. It is populated and then cleared on the next call.

**Action**: Remove `_reqid_to_mm_type` and the population code.

#### D3: `_process_text_image_output` — No Stage Config Uses "text+image" (lines 383-384, 406-424)

The routing branch `elif output_type in ("text+image", "text,image", "image+text")` and its handler `_process_text_image_output` are never triggered because:

- No stage config in the entire project uses `engine_output_type: text+image`, `text,image`, or `image+text`
- `OmniEngineCoreOutput` does not have an `output_type` attribute, so `getattr(eco, "output_type", ...)` always falls back to `engine_core_output_type`
- No code path ever sets `engine_core_output_type` to any of these composite strings

**Actual stage config values** (exhaustive search across all YAML files):
- `latent` — Qwen3-TTS, Qwen3-Omni, Qwen2.5-Omni, MiMo-Audio
- `audio` — Qwen3-TTS, Qwen3-Omni, Qwen2.5-Omni, MiMo-Audio
- `text` — Bagel
- `image` — Bagel
- `token_ids` — GLM-Image

**Action**: Remove `_process_text_image_output` and its routing branch.

#### D4: `"speech"` Alias — Never Used (line 387)

The routing matches `("audio", "speech")`, but no stage config ever uses `engine_output_type: speech`. Only `"audio"` is used in practice.

**Action**: Remove `"speech"` from the match (minor, but reduces confusion).

#### D5: `_extract_from_multimodal_outputs` — Always Returns None (lines 457-469)

This method tries to read `eco.multimodal_outputs`:

```python
mm = getattr(eco, "multimodal_outputs", None)
```

However, `OmniEngineCoreOutput` (defined in `vllm_omni/engine/__init__.py:76-77`) only has:

```python
class OmniEngineCoreOutput(EngineCoreOutput):
    pooling_output: dict[str, torch.Tensor] | None = None
```

There is **no `multimodal_outputs` attribute** on `EngineCoreOutput` or `OmniEngineCoreOutput`. The `getattr` always returns `None`, making this method always return `None`.

This means the "extract from multimodal_outputs" fallback inside `_process_image_output`, `_process_latents_output`, `_process_audio_output`, and `_process_text_image_output` is **dead code**. These methods only execute when `eco.pooling_output is None`, and in that case they try `_extract_from_multimodal_outputs` which always returns `None` — so they effectively do nothing.

**Action**: Remove `_extract_from_multimodal_outputs` and the extraction fallback paths in all `_process_*_output` methods.

#### D6: `_process_pooling_output` — Broken for Actual Input Type (lines 446-455)

This method is hit as a fallback when `output_type` is unrecognized (e.g., `"token_ids"` for GLM-Image). It tries:

```python
if not isinstance(eco.pooling_output, torch.Tensor):
    eco.pooling_output = torch.as_tensor(eco.pooling_output)
```

But `OmniEngineCoreOutput.pooling_output` is typed as `dict[str, torch.Tensor] | None`. Calling `torch.as_tensor()` on a dict will raise a `TypeError`. This method is either dead or broken.

**Action**: Remove or fix `_process_pooling_output`. For `"token_ids"`, the data already flows correctly through `pooling_output` → `add_multimodal_tensor` in `process_outputs`.

#### D7: `_process_text_output` — No-Op (lines 442-444)

```python
def _process_text_output(self, eco: EngineCoreOutput) -> None:
    """No-op; base processor will detokenize new_token_ids → text."""
    return
```

This is a no-op method. It adds no value and can be replaced by a direct `pass` or `return` in the routing.

**Action**: Remove `_process_text_output` and replace the routing branch with a simple return/pass.

#### D8: Key Remapping Logic — Obscure and Partially Redundant (lines 64-69)

In `add_multimodal_tensor`, the code remaps keys `"model_outputs"` and `"hidden"` to `target_key` (which is `mm_type`):

```python
if k == "model_outputs":
    k = target_key
elif k == "hidden" and target_key != "hidden":
    k = target_key
```

While this renaming does execute (AR runners produce `{"hidden": ...}` and generation runners produce `{"model_outputs": ...}`), downstream consumers like `serving_speech.py` still check for the **original key** `"model_outputs"`:

```python
key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
```

This means the renaming creates an inconsistency: the output processor renames `"model_outputs"` → `"audio"`, but the serving layer expects `"model_outputs"`. The system works only because the serving layer has a fallback, but the renaming adds confusion.

**Action**: Audit all downstream consumers and either commit to the renaming (update consumers) or remove it (keep original keys).

### 3.2 Effectively Unused Code Summary

| Code | Lines | Status | Reason |
|------|-------|--------|--------|
| `register_handler()` + `output_handlers` dict | 281-292, 277, 373-379 | **Dead** | Zero callers in entire codebase |
| `_reqid_to_mm_type` dict | 278, 346-350 | **Dead** | Populated but never read |
| `_process_text_image_output` | 383-384, 406-424 | **Dead** | No stage config uses "text+image"/"text,image"/"image+text" |
| `"speech"` alias in routing | 387 | **Dead** | No config uses "speech", only "audio" |
| `_extract_from_multimodal_outputs` | 457-469 | **Dead** | `eco.multimodal_outputs` attribute doesn't exist on `OmniEngineCoreOutput` |
| Extraction fallbacks in `_process_*_output` | 399-440 (partial) | **Dead** | All call `_extract_from_multimodal_outputs` which always returns None |
| `_process_pooling_output` | 446-455 | **Broken** | `torch.as_tensor()` on a dict raises TypeError |
| `_process_text_output` | 442-444 | **No-op** | Empty method body |
| Key remapping (`"model_outputs"` → mm_type) | 64-69 | **Inconsistent** | Renames keys that consumers still expect under original names |

**Estimated dead/broken code**: ~90 lines out of 469 (19% of the file).

### 3.3 What Actually Executes

After removing dead code, the **actual live data flow** is remarkably simple:

```
1. process_outputs() is called with a list of EngineCoreOutput
2. For each eco:
   a. engine_core_output_type determines mm_type (from stage config: "latent", "audio", "text", "image", "token_ids")
   b. _route_and_normalize() is called but effectively does nothing useful
      (all _process_*_output methods only act when pooling_output is None,
       then try _extract_from_multimodal_outputs which always returns None)
   c. If eco.pooling_output is not None AND req_state has a detokenizer:
      - pooling_output (a dict) is passed to req_state.add_multimodal_tensor()
      - eco.pooling_output is set to None (forcing text path in base processor)
3. Base vLLM OutputProcessor.process_outputs() handles text detokenization
4. On finish, _consolidate_multimodal_tensors() concatenates accumulated tensors
5. _new_completion_output() attaches mm_accumulated to CompletionOutput
```

The entire routing layer (`_route_and_normalize` and all `_process_*_output` methods) is effectively **a no-op** in production — the real work happens in the `pooling_output` capture at lines 355-361 and the accumulation in `OmniRequestState`.

## 4. Proposed Design

### 4.1 Core Abstractions

#### ModalityHandler (Protocol)

```python
class ModalityHandler(Protocol):
    """Handles extraction, normalization, and accumulation for a single output modality."""

    modality: str  # e.g., "image", "audio", "video", "latent"

    def extract(self, eco: EngineCoreOutput) -> torch.Tensor | dict | None:
        """Extract this modality's data from an EngineCoreOutput."""
        ...

    def accumulate(self, existing: Any, incoming: Any) -> Any:
        """Accumulate incremental outputs (e.g., streaming audio chunks)."""
        ...

    def consolidate(self, accumulated: Any) -> Any:
        """Finalize accumulated data (e.g., torch.cat for tensors)."""
        ...

    def validate(self, output: Any) -> bool:
        """Optional validation of the final output."""
        ...
```

#### ModalityRegistry

```python
class ModalityRegistry:
    """Central registry mapping modality names to their handlers."""

    def register(self, handler: ModalityHandler) -> None: ...
    def get(self, modality: str) -> ModalityHandler | None: ...
    def list_modalities(self) -> list[str]: ...
```

#### OutputRouter

```python
class OutputRouter:
    """Routes EngineCoreOutput to one or more ModalityHandlers."""

    def __init__(self, registry: ModalityRegistry): ...

    def route(self, eco: EngineCoreOutput, output_type: str) -> list[tuple[str, ModalityHandler]]:
        """Parse output_type and return list of (modality, handler) pairs.

        Supports composite types like "text+video+image" by splitting
        on '+' or ',' and looking up each component in the registry.
        """
        ...
```

### 4.2 Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MultimodalOutputProcessor (refactored)               │
│                                                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌───────────────────────┐  │
│  │ OutputRouter  │────▶│ ModalityReg  │────▶│ ModalityHandler[]     │  │
│  │              │     │              │     │  - TextHandler        │  │
│  │ "text+video" │     │ "text" → H1  │     │  - ImageHandler       │  │
│  │  ↓           │     │ "video" → H2 │     │  - VideoHandler       │  │
│  │ ["text",     │     │ "image" → H3 │     │  - AudioHandler       │  │
│  │  "video"]    │     │ "audio" → H4 │     │  - LatentHandler      │  │
│  └──────────────┘     │ "video" → H5 │     │  - (custom handlers)  │  │
│                       │ "latent"→ H6 │     └───────────────────────┘  │
│                       └──────────────┘                                 │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   OmniRequestState (refactored)                  │   │
│  │  mm_accumulators: dict[str, ModalityAccumulator]                 │   │
│  │  - Each accumulator uses its handler's accumulate/consolidate   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Built-in Handlers

| Handler | Modality | Extract Keys | Accumulation | Consolidation |
|---------|----------|-------------|--------------|---------------|
| `TextHandler` | `text` | (none — handled by base vLLM detokenizer) | N/A | N/A |
| `ImageHandler` | `image` | `image`, `images`, `pixel_values`, `pixels` | List append | `torch.cat(dim=0)` |
| `VideoHandler` | `video` | `video`, `videos`, `frames`, `video_frames` | List append | `torch.cat(dim=0)` or `torch.stack` |
| `AudioHandler` | `audio` | `audio`, `audios`, `wav`, `waveform`, `audio_pcm`, `pcm` | List append | `torch.cat(dim=-1)` (last dim for variable-length) |
| `LatentHandler` | `latent` | `latent`, `latents`, `z`, `posterior` | List append | `torch.cat(dim=0)` |

### 4.4 Composite Output Types

The `OutputRouter` supports composite types by splitting on delimiters:

```python
def route(self, eco, output_type):
    modalities = re.split(r'[+,]', output_type)
    handlers = []
    for m in modalities:
        m = m.strip().lower()
        handler = self.registry.get(m)
        if handler:
            handlers.append((m, handler))
        else:
            logger.warning("No handler for modality: %s", m)
    return handlers
```

This naturally supports:
- `"text+video+image"` → `[TextHandler, VideoHandler, ImageHandler]`
- `"text+audio"` → `[TextHandler, AudioHandler]`
- `"image"` → `[ImageHandler]`

### 4.5 Refactored OmniRequestState

```python
class OmniRequestState(RequestState):
    def __init__(self, *args, registry: ModalityRegistry, **kwargs):
        super().__init__(*args, **kwargs)
        self._registry = registry
        self._accumulators: dict[str, Any] = {}

    def add_output(self, modality: str, payload: Any) -> None:
        handler = self._registry.get(modality)
        if handler is None:
            return
        existing = self._accumulators.get(modality)
        self._accumulators[modality] = handler.accumulate(existing, payload)

    def consolidate(self) -> dict[str, Any]:
        result = {}
        for modality, data in self._accumulators.items():
            handler = self._registry.get(modality)
            if handler:
                result[modality] = handler.consolidate(data)
            else:
                result[modality] = data
        return result
```

### 4.6 Model Output Contract

To standardize the interface between models and the output processor, we propose that models producing multimodal outputs should return an `OmniOutput` (already defined in `model_executor/models/output_templates.py`) with a well-defined `multimodal_outputs` dict:

```python
@dataclass
class OmniOutput:
    text_hidden_states: torch.Tensor
    multimodal_outputs: dict[str, torch.Tensor | dict]
    # Keys should match registered modality names: "image", "video", "audio", etc.
```

## 5. Migration Plan

### Phase 0: Simplify — Remove Dead Code and Clarify Live Paths

This phase is a prerequisite for any further refactoring. It reduces the file from 469 lines to ~300 and makes the actual data flow visible.

**Step 0.1**: Remove dead code (all items from Section 3.2):

- Delete `register_handler()`, `self.output_handlers`, and the custom handler dispatch block in `_route_and_normalize`
- Delete `self._reqid_to_mm_type` and all code that populates it
- Delete `_process_text_image_output` and its routing branch
- Delete `_extract_from_multimodal_outputs` and all extraction fallback paths
- Delete `_process_pooling_output`
- Delete `_process_text_output`
- Remove `"speech"` from the audio routing match

**Step 0.2**: Simplify `_route_and_normalize` to reflect what actually executes:

```python
def _route_and_normalize(self, eco: EngineCoreOutput) -> None:
    # Currently a no-op for all production paths.
    # Routing is actually handled by the pooling_output capture in process_outputs().
    # Kept as an extension point for future modality-specific normalization.
    pass
```

Or, if we want to keep it as a dispatch point for future use, collapse into:

```python
def _route_and_normalize(self, eco: EngineCoreOutput) -> None:
    output_type = (self.engine_core_output_type or "").lower()
    if not output_type or output_type in ("text", "latent", "audio", "image", "token_ids"):
        return  # All handled by pooling_output capture in process_outputs()
```

**Step 0.3**: Audit and resolve the key remapping inconsistency (D8):

- Option A: Remove renaming, keep original keys (`"model_outputs"`, `"hidden"`), update the proposed ModalityHandler to standardize keys at extraction time
- Option B: Commit to renaming, update `serving_speech.py` and other consumers to use mm_type keys

**Step 0.4**: Simplify `make_request_output` — the current override largely duplicates base class logic for streaming/DELTA handling. If the only additions are `_consolidate_multimodal_tensors()` and `_new_completion_output()`, consider hooking these without copying the full method.

**Estimated result**: ~300 lines (from 469), with clear separation between "what the code does" and "extension points for the future".

### Phase 1: Introduce Abstractions (Non-Breaking)

1. Create `ModalityHandler` protocol and `ModalityRegistry` class
2. Implement built-in handlers (`TextHandler`, `ImageHandler`, `AudioHandler`, `LatentHandler`)
3. Create `OutputRouter` with delimiter-based composite type parsing
4. Place these in a new module: `vllm_omni/engine/output_handlers.py`

### Phase 2: Refactor MultimodalOutputProcessor

1. Replace the simplified `_route_and_normalize` with `OutputRouter.route()`
2. Introduce handler-based dispatch for `pooling_output` extraction and normalization
3. The existing `register_handler()` API concept is naturally replaced by `ModalityRegistry.register()`

### Phase 3: Refactor OmniRequestState

1. Replace `mm_accumulated` dict with per-modality accumulators
2. Replace `add_multimodal_tensor` with `add_output(modality, payload)`
3. Replace `_consolidate_multimodal_tensors` with handler-driven `consolidate()`

### Phase 4: Add New Modalities

1. Add `VideoHandler` for video output support
2. Test composite types: `text+video`, `text+image+video`, etc.
3. Integrate with upcoming DeepSeek model support

## 6. Backward Compatibility

- `register_handler()` was never called externally — removing it has zero backward compatibility impact
- Existing stage configs with `engine_output_type` strings (e.g., `"audio"`, `"latent"`, `"image"`) continue to work unchanged
- The `output_type` field on `EngineCoreOutput` is unchanged
- Phase 0 (dead code removal) is a pure deletion with no behavioral change, since none of the removed paths were reachable

## 7. Testing Strategy

| Test Type | Phase | Scope |
|-----------|-------|-------|
| Regression (existing) | Phase 0 | Run `test_outputs.py` and all E2E tests after dead code removal — must pass unchanged |
| Smoke test | Phase 0 | Verify Qwen3-TTS (audio), GLM-Image (token_ids), Bagel (text+image) pipelines still work |
| Unit tests | Phase 1 | Each `ModalityHandler` (extract, accumulate, consolidate) |
| Integration tests | Phase 2 | `OutputRouter` with composite types |
| E2E tests | Phase 2-3 | Full pipeline with existing models (Qwen3-TTS, GLM-Image, etc.) |
| New modality tests | Phase 4 | Video output handling, `text+video+image` combinations |

## 8. Open Questions

1. Should we support dynamic handler priority (e.g., if both `image` and `latent` are present, which takes precedence for `pooling_output`)?
2. Should the modality registry be global (singleton) or per-engine-instance?
3. How should error handling work when one handler in a composite route fails but others succeed?
4. Should we support async accumulation for streaming scenarios?
