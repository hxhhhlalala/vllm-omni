# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A4 MXFP4 (Microscaling FP4) online/offline quantization for diffusion transformers.

Architecture mirrors mxfp8_config.py:

  MXFPLinearMethodBase                   – platform-agnostic skeleton (imported from mxfp8_config)
    NPUMxfp4LinearMethod                 – NPU single-scale offline (W4A4 MXFP4)
      NPUMxfp4OnlineLinearMethod         – NPU single-scale online (BF16 → FP4)
    NPUMxfp4DualScaleLinearMethod        – NPU dual-scale offline (W4A4 MXFP4 DualScale)
      NPUMxfp4DualScaleOnlineLinearMethod– NPU dual-scale online (BF16 → FP4)

Key differences from MXFP8:

  1. Precision: float4_e2m1fn_x2 (FP4 packed, 2 values per element).
     npu_dynamic_mx_quant(x) without dst_type defaults to float4_e2m1fn_x2.

  2. Weight layout: stored as (N, K) — NOT pre-transposed.
     FP4 uses a packed format; transposing a packed tensor is not safely contiguous.
     Transpose is done inline in _quant_matmul via layer.weight.transpose(0, 1).

  3. GEMM signature: npu_quant_matmul requires explicit
     x1_dtype=float4_e2m1fn_x2 and x2_dtype=float4_e2m1fn_x2.

  Scale layout: (N, S/2, 2) — same reshape as MXFP8, also NOT pre-transposed;
  transposed inline in _quant_matmul.

Reference: MindIE-SD W4A4MXFP4QuantLinear (mindiesd/quantization/layer.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.mxfp8_config import (
    MXFPLinearMethodBase,
    _LazyWeightMixin,
    _get_torch_npu,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionMXFP4Config(QuantizationConfig):
    """W4A4 MXFP4 quantization config for diffusion transformers.

    Supports both online (BF16 checkpoint → quantize at load time) and offline
    (pre-quantized MXFP4 checkpoint) modes, mirroring DiffusionMXFP8Config.

    MX (microscaling) format: groups of 32 K-dimension elements share one
    float8_e8m0fnu exponent scale. Weight and activation are float4_e2m1fn_x2.
    """

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP4Config:
        quant_method = cls.get_from_keys_or(config, ["quant_method"], "")
        is_serialized = "mxfp4" in str(quant_method).lower()
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_mxfp4_serialized=is_serialized,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if current_omni_platform.is_npu():
                if self.is_checkpoint_mxfp4_serialized:
                    return NPUMxfp4LinearMethod(self)
                return NPUMxfp4OnlineLinearMethod(self)
            # Placeholder for future platforms: add `elif is_cuda(): ...` here.
            raise NotImplementedError(
                "DiffusionMXFP4Config (W4A4 MXFP4) is currently only supported "
                "on NPU (Ascend) platforms. CUDA support is not yet implemented."
            )
        return None


# ---------------------------------------------------------------------------
# NPU MXFP4 offline method (pre-quantized checkpoint)
# ---------------------------------------------------------------------------


class NPUMxfp4LinearMethod(MXFPLinearMethodBase):
    """NPU W4A4 MXFP4 offline linear method for pre-quantized checkpoints.

    Weight canonical layout after process_weights_after_loading:
      weight      : (N, K) in float4_e2m1fn_x2  — NOT pre-transposed (FP4 packed)
      weight_scale: (N, S/2, 2) in float8_e8m0fnu — NOT pre-transposed

    Both are transposed inline in _quant_matmul, unlike MXFP8 which pre-transposes.
    NPUMxfp4OnlineLinearMethod normalizes to the same layout so apply() is shared.
    """

    def __init__(self, quant_config: DiffusionMXFP4Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register weight and per-group MX scale for a pre-quantized checkpoint."""
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # Weight: BF16 placeholder; cast to float4_e2m1fn_x2 in process_weights.
        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Scale: (N, K//32) per-group MX scale; sharded along output (N) for TP.
        num_groups = (input_size_per_partition + 31) // 32
        scale = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, num_groups, dtype=torch.float32),
            input_dim=None,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Cast checkpoint weight to FP4 and normalize scale layout."""
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        torch_npu = _get_torch_npu()

        # NPU: cast to float4_e2m1fn_x2. Weight stays (N, K) — no pre-transpose.
        w = layer.weight
        if w.dtype != torch_npu.float4_e2m1fn_x2:
            w = torch_npu.npu_dtype_cast(w.npu(), torch_npu.float4_e2m1fn_x2)

        # Scale: (N, S) → (N, S/2, 2). Not pre-transposed; done inline in _quant_matmul.
        s = layer.weight_scale.to(torch_npu.float8_e8m0fnu)
        s = s.reshape(s.shape[0], -1, 2).contiguous()

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        layer._already_called_process_weights_after_loading = True

    # --- NPU MXFP4 ops — shared with online path via inheritance ---

    def _quantize_activation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # No dst_type: npu_dynamic_mx_quant defaults to float4_e2m1fn_x2.
        return _get_torch_npu().npu_dynamic_mx_quant(x)

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        torch_npu = _get_torch_npu()
        # NPU npu_quant_matmul requires bias in float32.
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        # FP4 differences vs FP8:
        #   weight (N,K) transposed inline → (K,N); scale (N,S/2,2) transposed inline → (S/2,N,2).
        #   x1_dtype / x2_dtype required — FP4 dtype not inferred from tensor dtype.
        return torch_npu.npu_quant_matmul(
            x_q,
            layer.weight.transpose(0, 1),        # (K, N) inline
            layer.weight_scale.transpose(0, 1),  # (S/2, N, 2) inline
            scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=ori_dtype,
            group_sizes=[1, 1, 32],
        )


# ---------------------------------------------------------------------------
# NPU MXFP4 online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class NPUMxfp4OnlineLinearMethod(_LazyWeightMixin, NPUMxfp4LinearMethod):
    """NPU W4A4 MXFP4 online linear method.

    MRO: NPUMxfp4OnlineLinearMethod → _LazyWeightMixin → NPUMxfp4LinearMethod
         → MXFPLinearMethodBase → LinearMethodBase

      create_weights  : _LazyWeightMixin          (meta device + patched loader)
      process_weights : NPUMxfp4OnlineLinearMethod (BF16 → FP4 + normalize)
      apply / ops     : NPUMxfp4LinearMethod / MXFPLinearMethodBase  (shared)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        torch_npu = _get_torch_npu()

        # Materialise weight if still on meta device (dummy-weight init path).
        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # NPU: quantize BF16/FP16 (N, K) → FP4. No dst_type → float4_e2m1fn_x2.
        weight_fp4, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(layer.weight)

        # Weight stays (N, K) — no pre-transpose for FP4 packed format.
        # Scale: (N, S) → (N, S/2, 2). Not pre-transposed; done inline in _quant_matmul.
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2).contiguous()

        replace_parameter(layer, "weight", weight_fp4)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._already_called_process_weights_after_loading = True


# ---------------------------------------------------------------------------
# NPU MXFP4 dual-scale offline method (W4A4_MXFP4_DUALSCALE checkpoint)
# ---------------------------------------------------------------------------


class NPUMxfp4DualScaleLinearMethod(MXFPLinearMethodBase):
    """NPU W4A4 MXFP4 dual-scale offline method for pre-quantized checkpoints.

    Checkpoint tensors and their canonical post-load shapes:
      weight            : (N, K) int8    – FP4 packed (2 values per byte)
      weight_scale      : (N, K//32) uint8 → reshaped to (N, K//64, 2)
      weight_dual_scale : (N, K//512, 1) float32 → transposed to (K//512, N)
      mul_scale         : (K,) float32   – per-input-channel activation pre-scale

    Forward:
      x_scaled                    = x * mul_scale
      x_q, l0_scale, l1_scale     = npu_dynamic_dual_level_mx_quant(x_scaled)
      output                      = npu_dual_level_quant_matmul(
                                      x_q, weight, l0_scale, weight_dual_scale,
                                      l1_scale, weight_scale)

    Reference: MindIE-SD W4A4MXFP4DualQuantLinear (mindiesd/quantization/layer.py).
    """

    def __init__(self, quant_config: Any) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # FP4 packed: 2 values per int8 byte → stored as int8 in checkpoint
        layer.register_parameter("weight", ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=torch.int8),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        ))

        # Fine scale: one uint8 exponent per group of 32 K elements
        num_groups_fine = (input_size_per_partition + 31) // 32
        layer.register_parameter("weight_scale", ModelWeightParameter(
            data=torch.empty(output_size_per_partition, num_groups_fine, dtype=torch.uint8),
            input_dim=None, output_dim=0, weight_loader=weight_loader,
        ))

        # Coarse scale: one float32 per group of 512 K elements.
        # Shape (N, K_coarse, 1) matches checkpoint layout exactly, avoiding the
        # shape-mismatch assert in linear.py:1344.
        num_groups_coarse = (input_size_per_partition + 511) // 512
        layer.register_parameter("weight_dual_scale", ModelWeightParameter(
            data=torch.empty(output_size_per_partition, num_groups_coarse, 1, dtype=torch.float32),
            input_dim=None, output_dim=0, weight_loader=weight_loader,
        ))

        layer.register_parameter("mul_scale", ModelWeightParameter(
            data=torch.empty(input_size_per_partition, dtype=torch.float32),
            input_dim=None, output_dim=None, weight_loader=weight_loader,
        ))

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        torch_npu = _get_torch_npu()

        # int8 (FP4 packed) → float4_e2m1fn_x2 dtype → NZ hardware format
        w = torch_npu.npu_dtype_cast(layer.weight.data.npu(), torch_npu.float4_e2m1fn_x2)
        w = torch_npu.npu_format_cast(w.view(torch.int8), 29, customize_dtype=torch.int8)

        # (N, K//32) uint8 → (N, K//64, 2) for npu_dual_level_quant_matmul
        s = layer.weight_scale.data.npu()
        s = s.reshape(s.shape[0], -1, 2).contiguous()

        # (N, K//512, 1) → squeeze → (N, K//512) → transpose → (K//512, N)
        ds = layer.weight_dual_scale.data.squeeze(-1).transpose(0, 1).npu().contiguous()

        ms = layer.mul_scale.data.view(-1).npu().contiguous()

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        replace_parameter(layer, "weight_dual_scale", ds)
        replace_parameter(layer, "mul_scale", ms)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])
        x = x * layer.mul_scale
        x_q, l0_scale, l1_scale = self._quantize_activation(x)
        output = self._quant_matmul(x_q, l0_scale, l1_scale, layer, bias, ori_dtype)
        return output.reshape(*ori_shape[:-1], -1)

    def _quantize_activation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _get_torch_npu().npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        l0_scale: torch.Tensor,
        l1_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        torch_npu = _get_torch_npu()
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        # weight_scale is (N, K//64, 2) — NOT transposed; operator expects this layout.
        # weight_dual_scale is (K_coarse, N).
        return torch_npu.npu_dual_level_quant_matmul(
            x_q,
            layer.weight,
            l0_scale,
            layer.weight_dual_scale,
            l1_scale,
            layer.weight_scale,
            bias=bias,
            output_dtype=ori_dtype,
        )


# ---------------------------------------------------------------------------
# NPU MXFP4 dual-scale online method (BF16 checkpoint → quantize at load time)
# ---------------------------------------------------------------------------


class NPUMxfp4DualScaleOnlineLinearMethod(_LazyWeightMixin, NPUMxfp4DualScaleLinearMethod):
    """NPU W4A4 MXFP4 dual-scale online method: quantises BF16 weights at load time.

    MRO: NPUMxfp4DualScaleOnlineLinearMethod → _LazyWeightMixin
         → NPUMxfp4DualScaleLinearMethod → MXFPLinearMethodBase

        create_weights         : _LazyWeightMixin         (meta device + patched loader)
        process_weights        : NPUMxfp4DualScaleOnlineLinearMethod (BF16 → FP4 + scales)
        apply / _quant_matmul  : NPUMxfp4DualScaleLinearMethod (shared with offline path)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        torch_npu = _get_torch_npu()

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # Quantize BF16 weight → FP4 + dual-level scales.
        # Returns: (weight_fp4, l0_scale[coarse], l1_scale[fine])
        weight_fp4, w_l0_scale, w_l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            layer.weight.data.npu(), smooth_scale=None
        )

        # NZ hardware format for the FP4 weight
        w = torch_npu.npu_format_cast(weight_fp4.view(torch.int8), 29, customize_dtype=torch.int8)

        # Fine scale: (N, K//32) → (N, K//64, 2)
        s = w_l1_scale.reshape(w_l1_scale.shape[0], -1, 2).contiguous()

        # Coarse scale: (N, K_coarse) or (N,) → (K_coarse, N)
        ds = w_l0_scale.reshape(w_l0_scale.shape[0], -1).transpose(0, 1).contiguous()

        # No offline calibration: identity pre-scale
        ms = torch.ones(layer.input_size_per_partition, dtype=torch.float32, device="npu")

        replace_parameter(layer, "weight", w)
        replace_parameter(layer, "weight_scale", s)
        replace_parameter(layer, "weight_dual_scale", ds)
        replace_parameter(layer, "mul_scale", ms)
        layer._already_called_process_weights_after_loading = True
