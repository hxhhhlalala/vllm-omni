# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed MXFP8 + MXFP4 DualScale quantization config for diffusion transformers.

Used when a checkpoint has MXFP8 for early blocks and W4A4_MXFP4_DUALSCALE for the rest.
Block-index dispatch requires that linear layers are constructed with a prefix of the form
"blocks.N.*", which is threaded through WanTransformerBlock in wan2_2_transformer.py.

Config injected by merge_mxfp4_dualscale_checkpoint.py:
    {
        "quant_method": "mixed_mxfp",
        "num_mxfp8_blocks": <N>,
        "is_checkpoint_serialized": true
    }
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.mxfp4_config import NPUMxfp4DualScaleLinearMethod, NPUMxfp4OnlineLinearMethod
from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod, NPUMxfp8OnlineLinearMethod

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_BLOCK_IDX_RE = re.compile(r"^blocks\.(\d+)\.")


def _parse_block_idx(prefix: str) -> int | None:
    """Extract block index from prefix like 'blocks.5.attn1.to_q'."""
    m = _BLOCK_IDX_RE.match(prefix)
    return int(m.group(1)) if m else None


class DiffusionMixedMXFPConfig(QuantizationConfig):
    """Mixed MXFP8 (blocks 0..num_mxfp8_blocks-1) + MXFP4 DualScale (remaining blocks).

    offline mode: is_checkpoint_serialized=True
      blocks 0..N-1  → NPUMxfp8LinearMethod
      blocks N..     → NPUMxfp4DualScaleLinearMethod

    online  mode: is_checkpoint_serialized=False
      blocks 0..N-1  → NPUMxfp8OnlineLinearMethod
      blocks N..     → NPUMxfp4OnlineLinearMethod
    """

    def __init__(
        self,
        num_mxfp8_blocks: int,
        is_checkpoint_serialized: bool = False,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_mxfp8_blocks = num_mxfp8_blocks
        self.is_checkpoint_serialized = is_checkpoint_serialized
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mixed_mxfp"

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
    def from_config(cls, config: dict[str, Any]) -> DiffusionMixedMXFPConfig:
        num_mxfp8_blocks = cls.get_from_keys_or(config, ["num_mxfp8_blocks"], 0)
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_serialized"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            num_mxfp8_blocks=num_mxfp8_blocks,
            is_checkpoint_serialized=is_serialized,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        if not current_omni_platform.is_npu():
            raise NotImplementedError(
                "DiffusionMixedMXFPConfig is currently only supported on NPU (Ascend) platforms."
            )

        block_idx = _parse_block_idx(prefix)
        in_mxfp8_range = block_idx is not None and block_idx < self.num_mxfp8_blocks

        if self.is_checkpoint_serialized:
            if in_mxfp8_range:
                return NPUMxfp8LinearMethod(self)
            return NPUMxfp4DualScaleLinearMethod(self)
        else:
            if in_mxfp8_range:
                return NPUMxfp8OnlineLinearMethod(self)
            return NPUMxfp4OnlineLinearMethod(self)
