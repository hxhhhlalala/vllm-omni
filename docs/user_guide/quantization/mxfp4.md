# W4A4 MXFP4 Quantization

## Overview

W4A4 MXFP4 (Microscaling FP4) quantizes both weights and activations to FP4
(`float4_e2m1fn_x2`, packed 2 values per byte) using the OCP MX format: groups
of 32 K-dimension elements share a single `float8_e8m0fnu` exponent scale.

This method supports two modes that differ significantly in scale structure and
checkpoint format:

| Mode | Scale structure | Description |
|------|----------------|-------------|
| **Online** | Single-scale (per-32 fine only) | BF16 weights are quantized to MXFP4 at load time — no pre-processing needed |
| **Offline** | Dual-scale (fine per-32 + coarse per-512 + per-channel smooth pre-scale) | msModelSlim-exported MXFP4 DualScale weights converted to diffusers format via preprocessing script — all scale tensors are loaded directly from the checkpoint |

!!! warning "Online ≠ Offline"
    Online mode uses a **single-scale** (`NPUMxfp4OnlineLinearMethod`): one
    `float8_e8m0fnu` exponent per 32 K elements, computed on the fly from the
    BF16 weight. Offline mode uses a **dual-scale** (`NPUMxfp4DualScaleLinearMethod`): a
    fine scale (per-32 K), a coarse scale (per-512 K), and a per-input-channel
    smooth pre-scale (`mul_scale`) produced by calibration. The two levels and
    the smooth pre-scale are all stored in the checkpoint; loading an offline
    checkpoint with the online method (or vice versa) will produce incorrect
    results.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ⭕ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ⭕ |
| NVIDIA Ampere GPU (SM 80+) | ⭕ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU (Atlas 950 A5) | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this guide.

## Model Type Support

### Diffusion Model (Wan2.2)

| Model | Mode | Notes |
|-------|------|-------|
| Wan2.2-T2V-A14B | Online + Offline | MoE cascade; quantizes two transformers (`transformer` + `transformer_2`); offline uses mixed MXFP8 (early blocks) + MXFP4 DualScale (remaining blocks) |
| Wan2.2-I2V-A14B | Online + Offline | MoE cascade; same mixed-precision scheme as T2V-A14B |
| Wan2.2-TI2V-5B | ❌ Not supported | Parameter count too small; W4A4 quantization causes unacceptable accuracy loss |

!!! note "Mixed MXFP8 + MXFP4 for cascade models"
    For the A14B cascade models, the offline checkpoint uses
    `quant_method: mxfp8_mxfp4_dualscale`: the first `num_mxfp8_blocks`
    transformer blocks are stored as MXFP8 (W8A8), and the remaining blocks as
    MXFP4 DualScale (W4A4). The split is recorded in the injected
    `quantization_config` and is transparent to the serving command.

!!! warning "TI2V-5B not supported"
    Wan2.2-TI2V-5B is excluded from W4A4 quantization. Its smaller parameter
    count makes it significantly more sensitive to 4-bit quantization noise,
    resulting in unacceptable accuracy loss. Use [MXFP8](mxfp8.md) for TI2V-5B.

## Configuration

### Online Mode

Online mode requires no pre-processing. vLLM-Omni quantizes BF16 weights to
MXFP4 at load time using `npu_dynamic_mx_quant`. A single block scale
(`float8_e8m0fnu`, one per 32 K elements) is computed on the fly; no
calibration `mul_scale` is available.

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="mxfp4")

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_video.py --model <your-model> --quantization mxfp4

# Online serving
vllm serve <your-model> --omni --quantization mxfp4
```

### Offline Mode (DualScale)

Offline mode loads a pre-quantized DualScale checkpoint from msModelSlim. A
preprocessing step converts the raw quantized output to the diffusers format
expected by vLLM-Omni and injects the quantization config into
`transformer/config.json` so that vLLM-Omni auto-detects the offline path
without a `--quantization` flag.

#### Checkpoint tensor layout

Each quantized linear layer stores four tensors:

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `weight` | `(N, K)` | float8_e4m3fn | FP4 packed (2 values per byte) |
| `weight_scale` | `(N, K//32)` | uint8 | Fine block scale (`float8_e8m0fnu` bit pattern) |
| `weight_dual_scale` | `(N, K//512, 1)` | float32 | Coarse block scale |
| `mul_scale` | `(K,)` | float32 | Per-input-channel smooth pre-scale (from calibration) |

#### Step 1 — Quantize with msModelSlim

```bash
msmodelslim quant \
  --model_path /path/to/Wan2.2-T2V-A14B-Diffusers \
  --save_path  /path/to/wan2_2_t2v_quantized_raw \
  --device npu \
  --model_type Wan2_2 \
  --config_path /path/to/wan2_2_w4a4_mxfp4_dualscale.yaml \
  --trust_remote_code True
```

After this step, `--save_path` contains the raw quantized safetensors files,
scale files, and a metadata JSON (`quant_model_description*.json`).

For cascade MoE models (T2V-A14B, I2V-A14B), msModelSlim outputs two
subdirectories: `high_noise_model/` and `low_noise_model/`.

#### Step 2 — Preprocess with merge_mxfp4_dualscale_checkpoint.py

The script (`vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py`):

1. Copies the original diffusers model to `--output-path` (VAE, text encoder,
   scheduler, etc. are preserved).
2. Remaps tensor names from msModelSlim convention to diffusers convention.
3. Saves the converted weights, fine/coarse scales, and `mul_scale` as
   `diffusion_pytorch_model.safetensors`.
4. Copies the original `transformer/config.json` and injects
   `quantization_config` so that vLLM-Omni auto-detects offline MXFP4
   DualScale.

For cascade MoE models, steps 2–4 run separately for `high_noise_model/` →
`transformer/` and `low_noise_model/` → `transformer_2/`.

```bash
python vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py \
  --model-type     Wan2.2-T2V-A14B \
  --original-model /path/to/Wan2.2-T2V-A14B-Diffusers \
  --quant-path     /path/to/wan2_2_t2v_quantized_raw \
  --output-path    /path/to/Wan2.2-T2V-A14B-MXFP4-DualScale
```

| Argument | Description |
|----------|-------------|
| `--model-type` | Model variant: `Wan2.2-T2V-A14B` or `Wan2.2-I2V-A14B` |
| `--original-model` | Root directory of the original BF16 diffusers model |
| `--quant-path` | Root directory of the msModelSlim quantized output |
| `--output-path` | Output directory for the merged model (created by the script) |

The script outputs a complete diffusers model directory at `--output-path`,
with each transformer subfolder containing:

- `diffusion_pytorch_model.safetensors` — converted FP4 weights, fine/coarse scales, and `mul_scale`
- `config.json` — original transformer config with `quantization_config` injected
- `quant_model_description.json` — renamed quantization metadata (reference only)

The `quantization_config` injected into `config.json` for each transformer:

```json
{
  "quant_method": "mxfp8_mxfp4_dualscale",
  "num_mxfp8_blocks": 5,
  "is_checkpoint_serialized": true
}
```

#### Step 3 — Serve

```bash
python text_to_video.py --model /path/to/Wan2.2-T2V-A14B-MXFP4-DualScale

# Online serving
vllm serve /path/to/Wan2.2-T2V-A14B-MXFP4-DualScale --omni
```

Python API:

```python
omni = Omni(model="/path/to/Wan2.2-T2V-A14B-MXFP4-DualScale")
```

!!! note
    No `--quantization` flag is needed for offline mode. The preprocessing
    script injects `quantization_config` into each `transformer/config.json`,
    which vLLM-Omni reads automatically to activate the offline MXFP4
    DualScale method.

## Parameters

### Online Mode (`mxfp4`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Must be `"mxfp4"` |
| `is_checkpoint_mxfp4_serialized` | bool | `False` | Set `True` to load a single-scale offline checkpoint; leave `False` (default) for online BF16-to-FP4 quantization |
| `ignored_layers` | list[str] | `[]` | Layer name substrings to keep in BF16 |

### Offline DualScale Mode (`mxfp8_mxfp4_dualscale`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Must be `"mxfp8_mxfp4_dualscale"` |
| `num_mxfp8_blocks` | int | `0` | Number of leading transformer blocks kept as MXFP8; remaining blocks use MXFP4 DualScale |
| `is_checkpoint_serialized` | bool | `False` | `True` for offline DualScale checkpoints; auto-set from `config.json` when using the preprocessing script |
| `ignored_layers` | list[str] | `[]` | Layer name substrings to keep in BF16 |

## Validation and Notes

1. **Online mode** quantizes BF16 weights at load time using
   `npu_dynamic_mx_quant` (single-scale). This adds a one-time overhead on the
   first load but requires no checkpoint preparation. No calibration
   `mul_scale` is available — all output partitions receive an identity
   pre-scale.

2. **Offline DualScale mode** loads four tensors per quantized layer: the FP4
   packed weight, a fine block scale (`uint8` interpreted as
   `float8_e8m0fnu`), a coarse block scale (`float32`), and a per-input-channel
   smooth pre-scale (`mul_scale`, `float32`). The `mul_scale` is derived from
   calibration and applied to the activation before dual-level quantization
   (`npu_dynamic_dual_level_mx_quant`), improving accuracy compared to the
   online single-scale path.

3. **Scale dtype**: fine scales are stored as `uint8` in safetensors (same bit
   layout as `float8_e8m0fnu`) and are reinterpreted at load time without a
   dtype conversion, avoiding a lossy float32 round-trip.

4. **Self-attention QKV fusion**: the Q, K, V projection weights are fused into
   a single `QKVParallelLinear` layer. Their `mul_scale` tensors are identical
   (all three projections share the same input), so the three sequential loads
   are idempotent.

5. W4A4 carries inherently higher quantization noise than W8A8 (16 vs 256
   quantization levels). The DualScale offline method mitigates this with
   calibrated `mul_scale` smooth quantization; online single-scale mode trades
   accuracy for the convenience of not requiring a pre-processed checkpoint.
