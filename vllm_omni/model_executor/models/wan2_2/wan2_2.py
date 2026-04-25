# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan2.2 T2V / I2V / TI2V pipeline adapters for vllm-omni OmniModelRegistry.

Registers three Wan2.2 variants so each can be launched as a standalone DiT
inference stage via OmniDiffusionConfig.  All three support:

  - Native BF16/FP16 float inference  (quantization_config=None)
  - W8A8 MXFP8 quantized inference    (quantization_config="mxfp8")
  - W8A8 INT8 quantized inference     (quantization_config="int8")
  - Any other method in vllm-omni's quantization factory

Usage (YAML stage config):
    pipeline_cls_name: Wan22T2VForVideoGeneration   # T2V
    pipeline_cls_name: Wan22I2VForVideoGeneration   # I2V
    pipeline_cls_name: Wan22TI2VForVideoGeneration  # TI2V (5B dense)
    model: /path/to/wan2.2-xxx
    quantization_config: mxfp8   # optional; null for FP inference
"""

from __future__ import annotations

import logging

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    create_transformer_from_config,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_ti2v import Wan22TI2VPipeline
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

logger = logging.getLogger(__name__)


def _make_quant_transformer(od_config, config: dict) -> WanTransformer3DModel:
    """Create a quantization-aware transformer from config + OmniDiffusionConfig."""
    quant_config = getattr(od_config, "quantization_config", None)
    if quant_config is not None:
        logger.info(
            "Wan2.2: creating transformer with quant_config=%s",
            type(quant_config).__name__,
        )
    return create_transformer_from_config(config, quant_config=quant_config)


class Wan22T2VForVideoGeneration(Wan22Pipeline):
    """Wan2.2 T2V (Text-to-Video) pipeline — supports FP and quantized inference.

    Registered in OmniModelRegistry as ``Wan22T2VForVideoGeneration``.
    Set ``pipeline_cls_name: Wan22T2VForVideoGeneration`` in the stage YAML.
    """

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        return _make_quant_transformer(self.od_config, config)


class Wan22I2VForVideoGeneration(Wan22I2VPipeline):
    """Wan2.2 I2V pipeline — supports FP and quantized (MXFP8/INT8) inference.

    Registered in OmniModelRegistry as ``Wan22I2VForVideoGeneration``.
    Set ``pipeline_cls_name: Wan22I2VForVideoGeneration`` in the stage YAML.
    """

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        return _make_quant_transformer(self.od_config, config)


class Wan22TI2VForVideoGeneration(Wan22TI2VPipeline):
    """Wan2.2 TI2V pipeline — supports FP and quantized (MXFP8/INT8) inference.

    TI2V is a dense 5B model that accepts both text and a conditioning image.
    Registered in OmniModelRegistry as ``Wan22TI2VForVideoGeneration``.
    Set ``pipeline_cls_name: Wan22TI2VForVideoGeneration`` in the stage YAML.
    """

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        return _make_quant_transformer(self.od_config, config)


__all__ = [
    "Wan22T2VForVideoGeneration",
    "Wan22I2VForVideoGeneration",
    "Wan22TI2VForVideoGeneration",
]
