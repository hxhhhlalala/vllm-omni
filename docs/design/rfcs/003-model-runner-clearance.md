# RFC-003: Model Runner Clearance — Extracting Model-Specific Logic

| Field        | Value                                |
|-------------|--------------------------------------|
| **Status**  | Draft                                |
| **Authors** | vLLM-Omni Team                       |
| **Created** | 2026-03-01                           |

## 1. Summary

This RFC proposes extracting model-specific logic currently embedded in the model runner classes (`OmniGPUModelRunner`, `GPUARModelRunner`, `GPUGenerationModelRunner`) into well-defined extension points and per-model adapter classes. The goal is to make the model runner a **model-agnostic execution framework** that delegates model-specific behavior through a clean plugin/adapter interface.

## 2. Motivation

### Current State

`OmniGPUModelRunner` (1,292 lines in `vllm_omni/worker/gpu_model_runner.py`) is the core execution class that inherits from `vllm.v1.worker.gpu_model_runner.GPUModelRunner`. Over time, it has accumulated significant model-specific logic that should not live in a generic runner:

#### M-RoPE / Positional Encoding (Qwen2-VL, HunYuan-VL, GLM-Image)

- `_init_mrope_positions()` (lines 99-157): Extracts `image_grid_thw`, `video_grid_thw`, `second_per_grid_ts`, `audio_feature_lengths`, `use_audio_in_video` — all Qwen-specific metadata.
- `_calc_mrope_positions()` (line 159): Delegates to upstream, then calls `_fixup_precomputed_mrope_decode_positions` for GLM-Image.
- `_fixup_precomputed_mrope_decode_positions()` (lines 180-222): Entirely GLM-Image-specific — handles 2D spatial decode positions for image generation models with `precomputed_mrope_decode = True`.
- XD-RoPE handling via `uses_xdrope_dim` for HunYuan-VL.

#### Talker MTP (Qwen3-TTS, Qwen3-Omni)

- `load_model()` (lines 71-97): Contains TTS-specific buffer allocation (`talker_mtp_input_ids`, `talker_mtp_inputs_embeds`, `last_talker_hidden`, `text_step`) with a `TODO` comment: "move this model specific logic to a separate class".
- `_talker_mtp_forward()` (lines 1215-1246): Entirely Qwen3-TTS/Omni-specific — runs the talker MTP forward pass with CUDA graph wrapping.
- `_preprocess()` (lines 1039-1213): Contains ~50 lines of talker MTP path (lines 1154-1204) interleaved with generic preprocessing.

#### MiMo-Audio

- `_maybe_attach_mimo_audio_req_infos()` (lines 1016-1037): Checks `self.model.__class__.__name__ == "MiMoAudioForConditionalGeneration"` — a direct class name check that couples the runner to a specific model.

#### Additional Information / Postprocess

- `_process_additional_information_updates()` (lines 937-969): Checks `self.model.has_postprocess` and calls `self.model.postprocess()` — a generic hook that's only used by specific models.
- `_preprocess()` similarly checks `self.model.has_preprocess` and calls `self.model.preprocess()`.

### Problems

| Problem | Impact |
|---------|--------|
| **Tight coupling** | Adding a new model with unique positional encoding or pre/post-processing requires modifying the runner itself |
| **Code readability** | The 1,292-line runner mixes framework logic with model-specific branches, making it hard to understand the core execution flow |
| **Testing difficulty** | Model-specific logic cannot be unit-tested in isolation from the full runner |
| **Rebase fragility** | Model-specific code in the runner is a frequent source of conflicts during upstream vLLM rebases |
| **Platform duplication** | NPU, XPU, and ROCm runners must replicate or work around model-specific logic from the GPU runner |

## 3. Proposed Design

### 3.1 Model Runner Adapter Protocol

```python
class ModelRunnerAdapter(Protocol):
    """Extension point for model-specific behavior in the runner."""

    def on_model_loaded(self, runner: "OmniGPUModelRunner", model: nn.Module) -> None:
        """Called after model is loaded. Allocate model-specific buffers."""
        ...

    def init_request_positions(
        self, runner: "OmniGPUModelRunner", req_state: CachedRequestState
    ) -> None:
        """Initialize custom positional encoding for a new request (e.g., M-RoPE, XD-RoPE)."""
        ...

    def fixup_decode_positions(
        self, runner: "OmniGPUModelRunner", scheduler_output: SchedulerOutput
    ) -> None:
        """Apply model-specific position fixups for decode (e.g., GLM-Image 2D positions)."""
        ...

    def preprocess(
        self,
        runner: "OmniGPUModelRunner",
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Model-specific preprocessing before forward pass."""
        ...

    def postprocess(
        self,
        runner: "OmniGPUModelRunner",
        hidden_states: torch.Tensor,
        multimodal_outputs: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Model-specific postprocessing after forward pass."""
        ...

    def extract_outputs(
        self,
        runner: "OmniGPUModelRunner",
        hidden_states: torch.Tensor | OmniOutput,
    ) -> tuple[torch.Tensor, dict]:
        """Extract text hidden states and multimodal outputs."""
        ...

    def enrich_request_info(
        self,
        runner: "OmniGPUModelRunner",
        req_id: str,
        req_state: CachedRequestState,
        req_infos: dict | None,
    ) -> dict | None:
        """Enrich per-request info before forward (e.g., MiMo-Audio mm_features)."""
        ...
```

### 3.2 Adapter Registry

```python
class ModelAdapterRegistry:
    """Maps model architecture names to their runner adapters."""

    _adapters: dict[str, type[ModelRunnerAdapter]] = {}

    @classmethod
    def register(cls, model_cls_name: str):
        def decorator(adapter_cls):
            cls._adapters[model_cls_name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def get_adapter(cls, model: nn.Module) -> ModelRunnerAdapter:
        name = model.__class__.__name__
        adapter_cls = cls._adapters.get(name, DefaultModelRunnerAdapter)
        return adapter_cls()
```

### 3.3 Concrete Adapters

#### Default Adapter

```python
class DefaultModelRunnerAdapter:
    """No-op adapter for models without special requirements."""
    def on_model_loaded(self, runner, model): pass
    def init_request_positions(self, runner, req_state): pass
    def fixup_decode_positions(self, runner, scheduler_output): pass
    def preprocess(self, runner, input_ids, inputs_embeds, so, nst):
        return input_ids, inputs_embeds, {}
    def postprocess(self, runner, hs, mm, so): pass
    def extract_outputs(self, runner, hs):
        return (hs, {}) if isinstance(hs, torch.Tensor) else (hs[0], {})
    def enrich_request_info(self, runner, req_id, req_state, req_infos):
        return req_infos
```

#### Qwen3-Omni / TTS Adapter

```python
@ModelAdapterRegistry.register("Qwen3OmniMoeForConditionalGeneration")
@ModelAdapterRegistry.register("Qwen3TTSForConditionalGeneration")
class Qwen3TalkerAdapter(DefaultModelRunnerAdapter):
    def on_model_loaded(self, runner, model):
        # Allocate talker_mtp buffers, CUDAGraphWrapper, etc.
        ...

    def init_request_positions(self, runner, req_state):
        # M-RoPE with image_grid_thw, video_grid_thw, audio_feature_lengths
        ...

    def preprocess(self, runner, input_ids, inputs_embeds, so, nst):
        # Talker MTP forward for decode requests
        ...
```

#### GLM-Image Adapter

```python
@ModelAdapterRegistry.register("GLMImageForConditionalGeneration")
class GLMImageAdapter(DefaultModelRunnerAdapter):
    def fixup_decode_positions(self, runner, scheduler_output):
        # 2D spatial decode position fixup for precomputed_mrope_decode
        ...
```

#### MiMo-Audio Adapter

```python
@ModelAdapterRegistry.register("MiMoAudioForConditionalGeneration")
class MiMoAudioAdapter(DefaultModelRunnerAdapter):
    def enrich_request_info(self, runner, req_id, req_state, req_infos):
        # Attach mm_features and req_id
        ...
```

### 3.4 Refactored OmniGPUModelRunner

After extraction, the runner becomes much simpler:

```python
class OmniGPUModelRunner(GPUModelRunner):
    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        self._adapter = ModelAdapterRegistry.get_adapter(self.model)
        self._adapter.on_model_loaded(self, self.model)

    def _update_states(self, scheduler_output):
        super()._update_states(scheduler_output)
        for req_id, req_state in ...:
            self._adapter.init_request_positions(self, req_state)

    def _calc_mrope_positions(self, scheduler_output):
        super()._calc_mrope_positions(scheduler_output)
        self._adapter.fixup_decode_positions(self, scheduler_output)

    def _preprocess(self, scheduler_output, num_input_tokens, ...):
        # Generic preprocessing
        ...
        input_ids, inputs_embeds, extra = self._adapter.preprocess(...)
        return ...

    def extract_multimodal_outputs(self, hidden_states):
        return self._adapter.extract_outputs(self, hidden_states)
```

### 3.5 File Structure

```
vllm_omni/worker/
├── gpu_model_runner.py          # Cleaned OmniGPUModelRunner (~600 lines)
├── gpu_ar_model_runner.py       # Unchanged
├── gpu_generation_model_runner.py # Unchanged
├── adapters/                    # NEW: Model-specific adapters
│   ├── __init__.py              # ModelAdapterRegistry
│   ├── base.py                  # ModelRunnerAdapter protocol + DefaultAdapter
│   ├── qwen3_talker.py          # Qwen3-Omni / TTS adapter
│   ├── glm_image.py             # GLM-Image adapter
│   ├── mimo_audio.py            # MiMo-Audio adapter
│   └── hunyuan_vl.py            # HunYuan-VL (XD-RoPE) adapter
```

## 4. Migration Plan

| Phase | Scope | Risk |
|-------|-------|------|
| Phase 1 | Define `ModelRunnerAdapter` protocol and `ModelAdapterRegistry` | Low — additive only |
| Phase 2 | Extract M-RoPE and positional encoding into adapters | Medium — touches core init path |
| Phase 3 | Extract talker MTP logic into `Qwen3TalkerAdapter` | Medium — performance-critical path |
| Phase 4 | Extract MiMo-Audio logic | Low — isolated code |
| Phase 5 | Extract GLM-Image precomputed position fixup | Low — isolated code |
| Phase 6 | Clean up `_preprocess` to use adapter dispatch | High — central hot path |
| Phase 7 | Port platform adapters (NPU, XPU) to use the same adapter system | Medium |

Each phase should be a separate PR with targeted tests.

## 5. Testing Strategy

| Test | Description |
|------|-------------|
| Adapter unit tests | Test each adapter's methods independently with mock runner |
| Registry tests | Verify correct adapter resolution by model class name |
| Integration tests | Run existing E2E tests (Qwen3-TTS, GLM-Image, MiMo-Audio) |
| Performance regression | Benchmark to ensure adapter indirection doesn't add measurable overhead |
| Platform compatibility | Verify NPU/XPU runners work with the adapter system |

## 6. Benefits

| Benefit | Impact |
|---------|--------|
| **Code clarity** | Runner reduced from ~1,300 to ~600 lines, with model logic isolated |
| **Extensibility** | New models only need an adapter class, not runner modifications |
| **Testability** | Model-specific logic can be unit-tested independently |
| **Rebase safety** | Upstream changes to `GPUModelRunner` are less likely to conflict with model-specific adapters |
| **Platform parity** | NPU/XPU runners can share adapters or provide platform-specific overrides |

## 7. Open Questions

1. Should adapters be auto-discovered from the model class or explicitly configured in stage configs?
2. Should the adapter have access to the full runner state or a restricted view (for safety)?
3. How do we handle models that need multiple adapter behaviors composed together?
4. Should `GPUARModelRunner` and `GPUGenerationModelRunner` also use the adapter pattern?
