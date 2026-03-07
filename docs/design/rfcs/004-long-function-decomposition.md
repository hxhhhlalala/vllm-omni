# RFC-004: Long Function Decomposition for Improved Readability

| Field        | Value                                |
|-------------|--------------------------------------|
| **Status**  | Draft                                |
| **Authors** | vLLM-Omni Team                       |
| **Created** | 2026-03-01                           |

## 1. Summary

This RFC proposes systematically decomposing the long functions in vllm-omni's engine and worker modules. Several functions span hundreds of lines, making them difficult to read, test, and maintain. We identify the worst offenders, propose decomposition strategies, and define guidelines to prevent future regression.

## 2. Motivation

### Identified Long Functions

A survey of the vllm-omni codebase reveals the following functions that significantly exceed reasonable length limits:

| Function | File | Lines | Description |
|----------|------|-------|-------------|
| `_stage_worker_async` | `entrypoints/omni_stage.py` | ~400 | Async stage worker: device setup, engine init, main loop, profiler handling, output processing |
| `_stage_worker` | `entrypoints/omni_stage.py` | ~400 | Sync stage worker: near-identical structure to async version |
| `handle_profiler_task_local` | `entrypoints/omni_stage.py` (nested) | ~270 | Profiler start/stop/dump handling for LLM and diffusion engines |
| `_dummy_run` | `worker/gpu_model_runner.py` | ~320 | CUDA graph warm-up/capture with multiple execution paths |
| `_preprocess` | `worker/gpu_model_runner.py` | ~175 | Input preparation with multiple branches for mm/embeds/text/preprocess |
| `process_inputs` | `engine/input_processor.py` | ~193 | Multimodal input processing with tokenization, embeddings, LoRA |
| `create_model_config` | `engine/arg_utils.py` | ~146 | Config construction (duplicated for sync/async) |
| `_sequential_init_lock` | `entrypoints/omni_stage.py` | ~146 | Lock management for sequential GPU initialization |
| `process_engine_inputs` | `entrypoints/omni_stage.py` (class method) | ~50 | Input wiring between stages |

### Problems Caused by Long Functions

| Problem | Impact |
|---------|--------|
| **Cognitive load** | Developers must hold 200-400 lines of context in their head to understand control flow |
| **Testing difficulty** | Cannot unit-test internal logic phases; must test the entire function end-to-end |
| **Merge conflicts** | Long functions increase the probability of merge conflicts during rebases |
| **Code duplication** | `_stage_worker` and `_stage_worker_async` are near-duplicates (~800 lines combined) with minor async/await differences |
| **Debugging difficulty** | Stack traces point to a single long function; hard to isolate which phase failed |

## 3. Decomposition Proposals

### 3.1 `_stage_worker` and `_stage_worker_async` (omni_stage.py)

These two functions share ~90% of their logic. The proposed decomposition:

#### Phase Extraction

```python
# New helper functions (shared between sync and async):

def _init_stage_environment(stage_payload: dict) -> StageContext:
    """Phase 1: Environment setup — ZMQ, device mapping, plugins."""
    ...

def _init_stage_engine(
    model: str, stage_payload: dict, stage_ctx: StageContext
) -> LLMEngine | OmniDiffusion:
    """Phase 2: Engine initialization with sequential lock."""
    ...

def _init_stage_connectors(
    stage_id: str, connectors_config: dict
) -> dict[tuple[str, str], OmniConnectorBase]:
    """Phase 3: Build OmniConnectors."""
    ...

def _handle_profiler_task(
    task_type: OmniStageTaskType, engine: Any, stage_id: str, stage_type: str
) -> dict:
    """Phase 4: Profiler start/stop/dump (extracted from nested function)."""
    ...

def _process_single_batch(
    batch: list[dict], engine: Any, stage_ctx: StageContext
) -> list[dict]:
    """Phase 5: Execute a single batch through the engine."""
    ...

def _format_and_send_outputs(
    outputs: list, out_q: Queue, stage_ctx: StageContext
) -> None:
    """Phase 6: Format outputs and send via IPC."""
    ...
```

#### Sync/Async Unification

```python
def _stage_worker(model, stage_payload, in_q, out_q, ...):
    ctx = _init_stage_environment(stage_payload)
    engine = _init_stage_engine(model, stage_payload, ctx)
    connectors = _init_stage_connectors(ctx.stage_id, ctx.connectors_config)

    while True:
        batch = _collect_batch(in_q, ctx.max_batch_size, ctx.batch_timeout)
        if _is_shutdown(batch):
            break
        if _is_profiler_task(batch):
            result = _handle_profiler_task(batch.task_type, engine, ...)
            out_q.put(result)
            continue
        outputs = _process_single_batch(batch, engine, ctx)
        _format_and_send_outputs(outputs, out_q, ctx)

async def _stage_worker_async(model, stage_payload, in_q, out_q, ...):
    ctx = _init_stage_environment(stage_payload)
    engine = _init_stage_engine(model, stage_payload, ctx)  # async variant
    connectors = _init_stage_connectors(ctx.stage_id, ctx.connectors_config)

    while True:
        batch = await _collect_batch_async(in_q, ...)
        # ... same structure, async versions of batch processing
```

### 3.2 `_dummy_run` (gpu_model_runner.py)

This function handles CUDA graph capture, warm-up, and profiling. Decompose into:

```python
def _dummy_run(self, num_tokens, ...):
    num_scheduled_tokens, attn_metadata = self._prepare_dummy_batch(num_tokens, ...)
    hidden_states = self._execute_dummy_forward(
        num_scheduled_tokens, attn_metadata, ...
    )
    self._run_dummy_drafter(num_tokens, is_graph_capturing, ...)
    self._finalize_dummy_run(hidden_states, num_scheduled_tokens, ...)

def _prepare_dummy_batch(self, num_tokens, ...):
    """Set up fake batch data, attention metadata, LoRA if needed."""
    ...

def _execute_dummy_forward(self, num_scheduled_tokens, attn_metadata, ...):
    """Run the model forward pass for CUDA graph capture or warm-up."""
    ...

def _run_dummy_drafter(self, num_tokens, is_graph_capturing, ...):
    """Run speculative decoding drafter dummy pass if configured."""
    ...

def _finalize_dummy_run(self, hidden_states, num_scheduled_tokens, ...):
    """Register NVTX hooks, run EPLB step, compute logit indices."""
    ...
```

### 3.3 `_preprocess` (gpu_model_runner.py)

The current function has 4 major branches. Decompose into:

```python
def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
    input_ids, inputs_embeds, model_kwargs, ec_connector_output = (
        self._prepare_model_inputs(scheduler_output, num_input_tokens)
    )
    positions = self._compute_positions(num_input_tokens)
    intermediate_tensors = self._prepare_intermediate_tensors(
        num_input_tokens, intermediate_tensors
    )
    self._apply_model_specific_preprocessing(
        scheduler_output, input_ids, inputs_embeds, model_kwargs
    )
    return input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output

def _prepare_model_inputs(self, scheduler_output, num_input_tokens):
    """Branch: mm inputs / prompt embeds / has_preprocess / text-only."""
    ...

def _compute_positions(self, num_input_tokens):
    """Select M-RoPE / XD-RoPE / standard positions."""
    ...

def _apply_model_specific_preprocessing(self, ...):
    """Overlay prompt_embeds, run has_preprocess, talker MTP."""
    ...
```

### 3.4 `process_inputs` (input_processor.py)

Decompose into phases:

```python
def process_inputs(self, request_id, prompt, params):
    preprocessed = self._preprocess_prompt(request_id, prompt, params)
    tokenized = self._tokenize_and_validate(preprocessed, params)
    mm_data = self._extract_multimodal_data(tokenized, params)
    prompt_embeds = self._extract_prompt_embeds(tokenized, params)
    additional_info = self._extract_additional_information(tokenized, params)
    return self._build_engine_core_request(
        request_id, tokenized, mm_data, prompt_embeds, additional_info, params
    )
```

### 3.5 `create_model_config` (arg_utils.py)

This function is duplicated between `OmniEngineArgs` and `AsyncOmniEngineArgs` (~260 lines combined). Decompose and deduplicate:

```python
class OmniEngineArgs:
    def create_model_config(self):
        return _create_model_config_common(self, is_async=False)

class AsyncOmniEngineArgs:
    def create_model_config(self):
        return _create_model_config_common(self, is_async=True)

def _create_model_config_common(args, is_async: bool):
    """Shared config construction logic."""
    base_config = _build_base_config(args)
    _apply_omni_overrides(base_config, args)
    _validate_config(base_config)
    return base_config
```

## 4. Guidelines for Future Code

To prevent regression, we propose the following guidelines:

### 4.1 Function Length Limits

| Severity | Line Count | Action |
|----------|-----------|--------|
| OK | < 50 lines | No action needed |
| Warning | 50-100 lines | Consider decomposition |
| Required | > 100 lines | Must decompose before merge |

### 4.2 Lint Rule

Add a custom lint check (or use `ruff` with `pylint.max-statements`) to flag functions exceeding the threshold:

```toml
# pyproject.toml
[tool.ruff.pylint]
max-statements = 50
```

### 4.3 Decomposition Principles

1. **Extract by phase**: Each phase of a multi-step function becomes its own method
2. **Extract by branch**: If a function has N major branches, each branch becomes a helper
3. **Extract by concern**: Separate I/O, computation, and error handling
4. **Preserve hot paths**: Do not add unnecessary function call overhead in performance-critical paths (profile after decomposition)
5. **Keep related logic together**: Extracted functions should be in the same module, not scattered

## 5. Implementation Plan

| Phase | Target | Functions | Est. Impact |
|-------|--------|-----------|-------------|
| Phase 1 | `omni_stage.py` | `_stage_worker`, `_stage_worker_async`, `handle_profiler_task_local` | ~800 lines → ~200 + shared helpers |
| Phase 2 | `gpu_model_runner.py` | `_dummy_run`, `_preprocess` | ~500 lines → ~200 + helpers |
| Phase 3 | `input_processor.py` | `process_inputs` | ~193 lines → ~80 + helpers |
| Phase 4 | `arg_utils.py` | `create_model_config` (dedup) | ~260 lines → ~130 shared |
| Phase 5 | Add lint rules | Configure ruff/pylint | Prevents regression |

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Performance regression** | Benchmark critical paths (model runner `_preprocess`, `_dummy_run`) before and after |
| **Behavioral changes** | Each decomposition should be a pure refactor — no logic changes |
| **Merge conflicts** | Coordinate with ongoing PRs; complete in small incremental PRs |
| **Variable scoping issues** | Use `dataclass` or `NamedTuple` for passing multi-value state between phases rather than long parameter lists |

## 7. Success Metrics

- No function in the codebase exceeds 100 lines (excluding comments/docstrings)
- `_stage_worker` and `_stage_worker_async` share > 80% of logic via shared helpers
- No performance regression in model runner benchmarks
- Improved code review velocity for PRs touching refactored files

## 8. Open Questions

1. Should we introduce a `StageContext` dataclass to pass state between phases in `_stage_worker`, or use simpler parameter passing?
2. Is the 100-line limit too aggressive for GPU kernel-related code paths where inlining matters?
3. Should we use `ruff` max-statements or a custom pre-commit hook for enforcement?
4. How do we handle the sync/async duplication — use a code generation approach or manual unification?
