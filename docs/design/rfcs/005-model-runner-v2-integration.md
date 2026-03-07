# RFC-005: Model Runner V2 Integration Roadmap

| Field        | Value                                |
|-------------|--------------------------------------|
| **Status**  | Draft                                |
| **Authors** | vLLM-Omni Team                       |
| **Created** | 2026-03-01                           |

## 1. Summary

Upstream vLLM is developing a V2 model runner to replace the current V1 architecture (`vllm.v1.worker.gpu_model_runner.GPUModelRunner`). vllm-omni currently **explicitly disables** V2 model runner support with a fallback warning. This RFC outlines the adaptation work required to integrate the V2 model runner into vllm-omni, including identifying the "omni hooks" that must be ported, defining a phased migration plan, and establishing compatibility testing.

## 2. Motivation

### Current State

Both `GPUARWorker` and `GPUGenerationWorker` contain identical guard code:

```python
# vllm_omni/worker/gpu_ar_worker.py:96-99
if self.use_v2_model_runner:
    logger.warning("OMNI GPUARWorker forces v1 model runner for omni hooks.")
    self.use_v2_model_runner = False

# vllm_omni/worker/gpu_generation_worker.py:96-99
if self.use_v2_model_runner:
    logger.warning("OMNI GPUGenerationWorker forces v1 model runner for omni hooks.")
    self.use_v2_model_runner = False
```

The scheduler already has partial V2 awareness:

```python
# vllm_omni/core/sched/output.py:44
prefill_token_ids: Optional  # prefill token IDs for v2 model runner
```

`OmniNewRequestData.from_request()` conditionally populates `prefill_token_ids` based on `use_v2_model_runner`.

### Why V2 Matters

| Reason | Detail |
|--------|--------|
| **Upstream convergence** | vLLM is moving toward V2 as the default; V1 may eventually be deprecated |
| **Performance** | V2 model runner introduces optimizations for attention metadata, micro-batching, and CUDA graph management |
| **Feature access** | New upstream features may only be available in V2 |
| **Maintenance burden** | Supporting V1 indefinitely means maintaining a growing delta from upstream |

### What Are "Omni Hooks"?

The warning message references "omni hooks" — the vllm-omni extensions to the model runner that V2 does not yet support. Based on code analysis, these are:

| Hook | Location | Description |
|------|----------|-------------|
| `additional_information` storage | `OmniGPUModelRunner.__init__` | Per-request `additional_information_cpu` attribute on `CachedRequestState` |
| `_decode_and_store_request_payloads` | `OmniGPUModelRunner` | Decodes prompt_embeds and additional_information from scheduler output for new requests |
| `_collect_additional_information_for_prefill` | `OmniGPUModelRunner` | Overlays prompt_embeds on inputs_embeds during prefill |
| `_build_model_kwargs_extra` | `OmniGPUModelRunner` | Injects `runtime_additional_information` into model forward kwargs |
| `_process_additional_information_updates` | `OmniGPUModelRunner` | Merges model postprocess outputs back into request state |
| `_merge_additional_information_update` | `OmniGPUModelRunner` | CPU-side merge of update dicts with tensor offloading |
| `extract_multimodal_outputs` | `OmniGPUModelRunner` | Splits `OmniOutput` into text hidden states + multimodal outputs |
| `_model_forward` override | `OmniGPUModelRunner` | Injects extra kwargs and caches model output for multimodal |
| M-RoPE extensions | `OmniGPUModelRunner` | Extended M-RoPE with audio, video, spatial metadata |
| XD-RoPE | `OmniGPUModelRunner` | HunYuan-VL dimensional RoPE |
| Talker MTP | `OmniGPUModelRunner` | Qwen3-TTS/Omni code predictor forward pass |
| `has_preprocess` / `has_postprocess` | `OmniGPUModelRunner._preprocess` | Per-request model preprocessing/postprocessing |
| `OmniModelRunnerOutput` | `outputs.py` | Extended `ModelRunnerOutput` with `multimodal_outputs` and `kv_extracted_req_ids` |

## 3. Analysis of V2 Model Runner Architecture

### 3.1 Key V2 Changes (Based on Upstream vLLM)

The V2 model runner refactors several areas that affect omni hooks:

| Area | V1 Behavior | V2 Behavior | Omni Impact |
|------|------------|-------------|-------------|
| **Input batch** | `GPUInputBatch` with direct attribute access | Restructured batch with new token layout | `_collect_additional_information_for_prefill` needs adaptation |
| **Forward context** | `set_forward_context()` with CUDA graph mode | Enhanced forward context with batch descriptors | `_model_forward` override must align |
| **Attention metadata** | Per-layer metadata builders | Restructured metadata pipeline | M-RoPE position injection path changes |
| **Output handling** | `ModelRunnerOutput` returned from `execute_model` | Potentially different output structure | `OmniModelRunnerOutput` needs remapping |
| **CUDA graph capture** | `_dummy_run` with manual graph management | V2 CUDA graph system | `_dummy_run` override may need rewrite |
| **Scheduler output** | `SchedulerOutput` with `scheduled_new_reqs` | V2 scheduler output format | Request payload decoding path changes |

### 3.2 Hook Compatibility Assessment

| Hook | V2 Compatibility | Effort |
|------|-----------------|--------|
| `additional_information` storage | **Needs adaptation** — V2 may change `CachedRequestState` | Medium |
| `_decode_and_store_request_payloads` | **Needs adaptation** — scheduler output format may differ | Medium |
| `_collect_additional_information_for_prefill` | **Needs adaptation** — input batch layout changes | High |
| `_build_model_kwargs_extra` | **Likely compatible** — model forward kwargs injection | Low |
| `_model_forward` override | **Needs adaptation** — forward context API changes | Medium |
| `extract_multimodal_outputs` | **Likely compatible** — output extraction is model-level | Low |
| M-RoPE extensions | **Needs adaptation** — position computation pipeline changes | High |
| Talker MTP | **Needs adaptation** — CUDA graph management changes | High |
| `OmniModelRunnerOutput` | **Needs adaptation** — output class hierarchy may change | Medium |

## 4. Proposed Integration Plan

### 4.1 Phase 0: Upstream Tracking (Ongoing)

- Monitor upstream V2 model runner PRs and API stabilization
- Maintain a mapping document of V1 → V2 API changes relevant to omni hooks
- Track which omni hooks have upstream equivalents in V2

### 4.2 Phase 1: Abstract the Hook Interface

Before integrating V2, abstract the hook points so that both V1 and V2 can be supported:

```python
class OmniModelRunnerHooks(Protocol):
    """Interface for omni extensions to the model runner."""

    def on_new_requests(self, scheduler_output) -> None:
        """Decode and store payloads for new requests."""
        ...

    def prepare_prefill_embeds(self, num_scheduled_tokens_np) -> None:
        """Overlay prompt_embeds for prefill."""
        ...

    def build_extra_kwargs(self) -> dict:
        """Build extra model forward kwargs."""
        ...

    def process_model_output(self, hidden_states, mm_outputs, scheduler_output) -> None:
        """Post-forward processing of model outputs."""
        ...

    def get_model_runner_output(self, base_output) -> ModelRunnerOutput:
        """Wrap base output with omni-specific fields."""
        ...
```

This abstraction aligns with RFC-003 (Model Runner Clearance) and can be implemented concurrently.

### 4.3 Phase 2: V2 Runner Skeleton

Create a V2-compatible runner class:

```python
class OmniGPUModelRunnerV2(GPUModelRunnerV2):
    """V2 model runner with omni hooks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hooks = OmniModelRunnerHooksImpl(self)

    def execute_model(self, scheduler_output, ...):
        self._hooks.on_new_requests(scheduler_output)
        self._hooks.prepare_prefill_embeds(...)
        result = super().execute_model(scheduler_output, ...)
        self._hooks.process_model_output(result, ...)
        return self._hooks.get_model_runner_output(result)
```

### 4.4 Phase 3: Port Individual Hooks

Port each hook to V2, prioritized by complexity and importance:

| Priority | Hook | Rationale |
|----------|------|-----------|
| P0 | `_model_forward` override | Core execution path |
| P0 | `extract_multimodal_outputs` | Required for any multimodal output |
| P0 | `OmniModelRunnerOutput` | Required for output pipeline |
| P1 | `additional_information` lifecycle | Required for multi-stage pipelines |
| P1 | M-RoPE extensions | Required for Qwen2-VL, Qwen3-Omni |
| P2 | Talker MTP | Required for Qwen3-TTS |
| P2 | `has_preprocess` / `has_postprocess` | Required for custom models |
| P3 | XD-RoPE | Only needed for HunYuan-VL |

### 4.5 Phase 4: Worker Updates

Update workers to conditionally use V2:

```python
class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    def init_device(self):
        ...
        if self.use_v2_model_runner:
            self.model_runner = GPUARModelRunnerV2(self.vllm_config, self.device)
        else:
            self.model_runner = GPUARModelRunner(self.vllm_config, self.device)
```

### 4.6 Phase 5: Validation and Cutover

1. Run full test suite with `use_v2_model_runner=True`
2. Benchmark V2 against V1 for key models (Qwen3-Omni, Qwen3-TTS, GLM-Image)
3. Enable V2 as opt-in via environment variable
4. Eventually make V2 the default when all hooks are ported

## 5. Compatibility Strategy

### 5.1 Dual Runner Support

During the transition, both V1 and V2 runners coexist:

```
vllm_omni/worker/
├── gpu_model_runner.py        # V1: OmniGPUModelRunner (existing)
├── gpu_model_runner_v2.py     # V2: OmniGPUModelRunnerV2 (new)
├── gpu_ar_model_runner.py     # V1 AR runner
├── gpu_ar_model_runner_v2.py  # V2 AR runner (new)
├── gpu_generation_model_runner.py     # V1 generation runner
├── gpu_generation_model_runner_v2.py  # V2 generation runner (new)
```

### 5.2 Feature Parity Matrix

Track feature parity with a compatibility matrix:

| Feature | V1 | V2 | Notes |
|---------|----|----|-------|
| Text generation | Yes | Phase 2 | |
| Multimodal input (image/video/audio) | Yes | Phase 2 | |
| Multimodal output | Yes | Phase 3 | |
| Multi-stage pipeline | Yes | Phase 3 | |
| M-RoPE | Yes | Phase 3 | |
| Talker MTP | Yes | Phase 3 | |
| CUDA graph capture | Yes | Phase 2 | |
| Prompt embeds | Yes | Phase 3 | |
| Diffusion models | Yes | Phase 4 | Separate runner |
| NPU support | Yes | Phase 5 | Depends on `vllm_ascend` V2 |
| XPU support | Yes | Phase 5 | |

## 6. Platform Considerations

### 6.1 NPU (Ascend)

NPU currently uses `vllm_ascend.worker.model_runner_v1.NPUModelRunner` as its base. The V2 integration depends on:
- Whether `vllm_ascend` provides a V2 model runner
- Whether NPU-specific changes can be layered on top of `OmniGPUModelRunnerV2`

### 6.2 XPU (Intel)

XPU runners (`XPUARModelRunner`, `XPUGenerationModelRunner`) inherit from the GPU runners. They should automatically benefit from V2 if the GPU V2 runner is correct, but may need platform-specific overrides for XPU memory management.

## 7. Testing Plan

| Level | Description |
|-------|-------------|
| Unit | Test each ported hook independently with mock V2 runner |
| Integration | Run E2E tests for each model with `use_v2_model_runner=True` |
| Benchmark | Compare V1 vs V2 throughput/latency for Qwen3-Omni, Qwen3-TTS |
| CI gate | Add V2 runner to CI matrix (opt-in initially, required later) |
| Platform | Test NPU/XPU with V2 runner (dependent on platform V2 support) |

## 8. Timeline

| Phase | Scope | Dependencies | Est. Duration |
|-------|-------|-------------|---------------|
| Phase 0 | Upstream tracking | None | Ongoing |
| Phase 1 | Abstract hook interface | RFC-003 | 2 weeks |
| Phase 2 | V2 runner skeleton + core hooks | Phase 1 + upstream V2 stabilization | 3 weeks |
| Phase 3 | Port all hooks | Phase 2 | 4 weeks |
| Phase 4 | Worker updates + opt-in flag | Phase 3 | 1 week |
| Phase 5 | Validation + platform ports | Phase 4 | 3 weeks |
| Cutover | V2 as default | Phase 5 + confidence from CI | TBD |

## 9. Risks

| Risk | Mitigation |
|------|------------|
| Upstream V2 API instability | Phase 0 tracking; abstract hook interface insulates from changes |
| Performance regression in V2 | Benchmark gating before cutover |
| Feature incompleteness | Feature parity matrix; V1 fallback always available |
| Platform support gaps | NPU/XPU V2 treated as separate Phase 5 work |
| Large merge conflicts during V2 port | Coordinate with rebase automation (RFC-001) |

## 10. Open Questions

1. Is upstream V2 model runner stable enough to begin Phase 2, or should we wait for an official release?
2. Should we attempt to contribute omni hooks upstream to V2, or keep them in vllm-omni?
3. How does V2 affect the `DiffusionModelRunner` (which has its own independent runner implementation)?
4. Should the V2 integration be gated behind a configuration flag or environment variable during the transition?
5. What is the minimum set of hooks needed for a useful V2 MVP (i.e., which models can work with a partial port)?
