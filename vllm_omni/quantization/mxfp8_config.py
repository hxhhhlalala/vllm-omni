# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 MXFP8 (Microscaling FP8) online quantization for diffusion transformers.

W8: weight quantized to FP8 (float8_e4m3fn) with MX group scales (E8M0 per 32 elements).
A8: activation quantized to FP8 (float8_e4m3fn) dynamically per-token each forward pass.

Only online quantization is supported: loads BF16/FP16 checkpoint and quantizes
weight to FP8 at load time via npu_dynamic_mx_quant.

Platform: NPU (Ascend) only.  Requires torch_npu with npu_dynamic_mx_quant and
npu_quant_matmul (MX-scale variant with scale_dtype=float8_e8m0fnu).

Reference: MindIE-SD W8A8MXFP8QuantLinear (mindiesd/quantization/layer.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

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
from vllm.model_executor.layers.quantization.fp8 import CopyNumelCounter, _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# Lazily imported to avoid hard dependency on NPU at module load time.
_torch_npu = None


def _get_torch_npu():
    global _torch_npu
    if _torch_npu is None:
        try:
            import torch_npu as _tnpu
            _torch_npu = _tnpu
        except ImportError as e:
            raise ImportError(
                "DiffusionMXFP8Config requires torch_npu. "
                "Please install the Ascend NPU toolkit."
            ) from e
    return _torch_npu


class DiffusionMXFP8Config(QuantizationConfig):
    """W8A8 MXFP8 online quantization config for NPU diffusion transformers.

    Loads BF16/FP16 checkpoints and quantizes linear layer weights to FP8 at
    load time using torch_npu.npu_dynamic_mx_quant.  Activations are quantized
    dynamically to FP8 on every forward pass.

    MX (microscaling) format: groups of 32 K-dimension elements share one
    float8_e8m0fnu exponent scale, matching MXFP8 as defined in the OCP MX spec.
    """

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Verified on Ascend A2/A3.  Not applicable to CUDA (NPU-only).
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper") -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionMXFP8Config":
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(ignored_layers=ignored_layers)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not current_omni_platform.is_npu():
                raise NotImplementedError(
                    "DiffusionMXFP8Config (W8A8 MXFP8) is currently only supported "
                    "on NPU (Ascend) platforms. CUDA support is not yet implemented."
                )
            return NPUMXFP8OnlineLinearMethod(self)
        return None


# ---------------------------------------------------------------------------
# Lazy-weight mixin (copied from int8_config.py; shared pattern)
# ---------------------------------------------------------------------------

class _LazyWeightMixin:
    """Weight registered on meta device, materialised just-in-time on first load.

    This is a local copy of the pattern in int8_config.LazyWeightMixin so that
    mxfp8_config.py has no import dependency on int8_config.py.
    """

    uses_meta_device: bool = True

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

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0
                # Materialise weight from meta device on first chunk.
                weight = ModelWeightParameter(
                    data=torch.empty_like(layer.weight, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=patched_weight_loader,
                )
                _copy_missing_attrs(layer.weight, weight)
                layer.register_parameter("weight", weight)
                del layer._load_device

            param = layer.weight
            counter = CopyNumelCounter()
            with counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)
            layer._loaded_numel += counter.copied_numel

            if layer._loaded_numel == layer.weight.numel():
                self.process_weights_after_loading(layer)
                layer._already_called_process_weights_after_loading = True

            return res

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=patched_weight_loader,
        )
        layer._load_device = torch.get_default_device()
        layer.register_parameter("weight", weight)


# ---------------------------------------------------------------------------
# NPU W8A8 MXFP8 online linear method
# ---------------------------------------------------------------------------

class NPUMXFP8OnlineLinearMethod(_LazyWeightMixin, LinearMethodBase):
    """NPU online W8A8 MXFP8 linear quantization method.

    Weight quantization (at load time, process_weights_after_loading):
      BF16/FP16 weight (N, K)
        → npu_dynamic_mx_quant(..., dst_type=float8_e4m3fn)
        → weight_fp8 (N, K) in float8_e4m3fn
        → transposed to (K, N), stored on layer
      weight_scale (N, S) in float8_e8m0fnu
        → reshape to (N, S/2, 2)          # MindIE-SD offline format
        → transposed to (S/2, N, 2)        # GEMM layout
        → stored on layer as weight_scale

    Activation quantization (each forward pass, apply):
      BF16/FP16 activation (M, K)
        → npu_dynamic_mx_quant(..., dst_type=float8_e4m3fn)
        → x_fp8 (M, K), x_scale per-token in float8_e8m0fnu

    GEMM:
      npu_quant_matmul(x_fp8, weight_fp8, weight_scale,
                       scale_dtype=float8_e8m0fnu,
                       pertoken_scale=x_scale,
                       pertoken_scale_dtype=float8_e8m0fnu,
                       group_sizes=[1, 1, 32])

    Note on weight_scale shape: npu_dynamic_mx_quant returns the weight scale
    in a shape that may vary across torch_npu versions.  The reshape and transpose
    applied here follow the MindIE-SD W8A8MXFP8QuantLinear convention and assume
    the returned scale has an even-sized last dimension.  Verify on your target
    torch_npu version if you encounter shape errors.
    """

    def __init__(self, quant_config: DiffusionMXFP8Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

    # create_weights is inherited from _LazyWeightMixin.

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        torch_npu = _get_torch_npu()

        # Materialise weight if it is still on the meta device (e.g. when no
        # real checkpoint shards were loaded, as in dummy-weight initialisation).
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

        # -------------------------------------------------------------------
        # Quantize weight: BF16/FP16 (N, K) → FP8 (N, K) + MX scale
        # -------------------------------------------------------------------
        weight = layer.weight  # (output, input) = (N, K)

        weight_fp8, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(
            weight, dst_type=torch_npu.float8_e4m3fn
        )
        # weight_fp8      : (N, K) in float8_e4m3fn
        # weight_scale_raw: (N, S) in float8_e8m0fnu
        #   S depends on the torch_npu version and group_size (typically K/16 or K/32).
        #   The reshape below follows MindIE-SD's offline weight_scale format.

        # Reshape scale to (N, S/2, 2) then pre-transpose to (S/2, N, 2)
        # so that apply() can pass it directly to npu_quant_matmul without
        # a per-forward-pass transpose.
        weight_scale = (
            weight_scale_raw
            .reshape(weight_scale_raw.shape[0], -1, 2)  # (N, S/2, 2)
            .transpose(0, 1)                             # (S/2, N, 2)
            .contiguous()
        )

        # Pre-transpose weight to (K, N) for GEMM (avoids per-forward transpose).
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous()  # (K, N)

        replace_parameter(layer, "weight", weight_fp8)
        # weight_scale is new (not registered in create_weights); replace_parameter
        # will register it as a buffer via module.register_buffer.
        replace_parameter(layer, "weight_scale", weight_scale)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        torch_npu = _get_torch_npu()

        ori_shape = x.shape
        ori_dtype = x.dtype

        # Flatten to 2-D for the NPU kernel (avoids per-element overhead on N-D
        # tensors, mirroring the _flatten_linear pattern in MindIE-SD).
        x = x.reshape(-1, ori_shape[-1])  # (M, K)

        # Dynamic MXFP8 quantisation of activation.
        x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float8_e4m3fn
        )
        # x_fp8  : (M, K) in float8_e4m3fn
        # x_scale: per-token MX scale in float8_e8m0fnu

        # npu_quant_matmul requires bias in float32 when provided.
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        # W8A8 MXFP8 fused GEMM.
        # layer.weight      : (K, N)      float8_e4m3fn  (pre-transposed)
        # layer.weight_scale: (S/2, N, 2) float8_e8m0fnu (pre-transposed)
        # x_scale           : per-token   float8_e8m0fnu
        output = torch_npu.npu_quant_matmul(
            x_fp8,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=ori_dtype,
            group_sizes=[1, 1, 32],
        )

        return output.reshape(*ori_shape[:-1], -1)
