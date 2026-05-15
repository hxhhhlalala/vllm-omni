# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 quantization configs and the mixed MXFP8+MXFP4 config."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# DiffusionMXFP4Config
# ---------------------------------------------------------------------------


def test_mxfp4_config_get_name():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    assert DiffusionMXFP4Config.get_name() == "mxfp4"


def test_mxfp4_config_from_config_defaults():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({})
    assert cfg.is_checkpoint_mxfp4_serialized is False
    assert cfg.ignored_layers == []


def test_mxfp4_config_from_config_serialized():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"is_checkpoint_mxfp4_serialized": True})
    assert cfg.is_checkpoint_mxfp4_serialized is True


def test_mxfp4_config_from_config_ignored_layers():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"ignored_layers": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


def test_mxfp4_config_from_config_modules_to_not_convert_fallback():
    """modules_to_not_convert must be accepted as an alias for ignored_layers."""
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = DiffusionMXFP4Config.from_config({"modules_to_not_convert": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


# ---------------------------------------------------------------------------
# DiffusionMXFP8MXFP4DualScaleConfig
# ---------------------------------------------------------------------------


def test_mixed_config_get_name():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    assert DiffusionMXFP8MXFP4DualScaleConfig.get_name() == "mxfp8_mxfp4_dualscale"


def test_mixed_config_no_args_does_not_raise():
    """DiffusionMXFP8MXFP4DualScaleConfig() with no args must not raise TypeError."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    cfg = DiffusionMXFP8MXFP4DualScaleConfig()
    assert cfg.num_mxfp8_blocks == 0
    assert cfg.is_checkpoint_serialized is False
    assert cfg.ignored_layers == []


def test_mixed_config_from_config_with_num_blocks():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    cfg = DiffusionMXFP8MXFP4DualScaleConfig.from_config(
        {
            "quant_method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 5,
            "is_checkpoint_serialized": True,
        }
    )
    assert cfg.num_mxfp8_blocks == 5
    assert cfg.is_checkpoint_serialized is True
    assert cfg.ignored_layers == []


def test_mixed_config_from_config_defaults():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    cfg = DiffusionMXFP8MXFP4DualScaleConfig.from_config({})
    assert cfg.num_mxfp8_blocks == 0
    assert cfg.is_checkpoint_serialized is False


def test_mixed_config_from_config_ignored_layers():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    cfg = DiffusionMXFP8MXFP4DualScaleConfig.from_config({"num_mxfp8_blocks": 3, "ignored_layers": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


# ---------------------------------------------------------------------------
# build_quant_config integration
# ---------------------------------------------------------------------------


def test_build_quant_config_mxfp4_string():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = build_quant_config("mxfp4")
    assert isinstance(cfg, DiffusionMXFP4Config)
    assert cfg.get_name() == "mxfp4"
    assert cfg.is_checkpoint_mxfp4_serialized is False


def test_build_quant_config_mxfp4_dict():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    cfg = build_quant_config({"method": "mxfp4", "is_checkpoint_mxfp4_serialized": True})
    assert isinstance(cfg, DiffusionMXFP4Config)
    assert cfg.is_checkpoint_mxfp4_serialized is True


def test_build_quant_config_mxfp8_mxfp4_dualscale_dict():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    cfg = build_quant_config(
        {
            "method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 5,
            "is_checkpoint_serialized": True,
        }
    )
    assert isinstance(cfg, DiffusionMXFP8MXFP4DualScaleConfig)
    assert cfg.num_mxfp8_blocks == 5
    assert cfg.is_checkpoint_serialized is True


def test_build_quant_config_mxfp8_mxfp4_dualscale_warns_without_num_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """build_quant_config('mxfp8_mxfp4_dualscale') must emit WARNING when
    num_mxfp8_blocks is absent and default to 0 (all-MXFP4 DualScale mode).

    Uses monkeypatch instead of caplog because vllm's init_logger may configure
    propagation in a way that prevents caplog from intercepting the messages.
    """
    import vllm_omni.quantization.factory as factory_module
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    warning_messages: list[str] = []
    monkeypatch.setattr(
        factory_module.logger,
        "warning",
        lambda msg, *args, **kw: warning_messages.append(msg),
    )

    cfg = build_quant_config("mxfp8_mxfp4_dualscale")

    assert isinstance(cfg, DiffusionMXFP8MXFP4DualScaleConfig)
    assert cfg.num_mxfp8_blocks == 0
    assert len(warning_messages) >= 1
    assert any("num_mxfp8_blocks" in msg for msg in warning_messages)


def test_build_quant_config_mxfp8_mxfp4_dualscale_info_with_num_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """When num_mxfp8_blocks is provided, INFO is logged and no WARNING is emitted."""
    import vllm_omni.quantization.factory as factory_module
    from vllm_omni.quantization import build_quant_config

    warning_messages: list[str] = []
    info_messages: list[str] = []
    monkeypatch.setattr(
        factory_module.logger,
        "warning",
        lambda msg, *args, **kw: warning_messages.append(msg),
    )
    monkeypatch.setattr(
        factory_module.logger,
        "info",
        lambda msg, *args, **kw: info_messages.append(msg),
    )

    cfg = build_quant_config(
        {
            "method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 5,
            "is_checkpoint_serialized": True,
        }
    )

    assert cfg.num_mxfp8_blocks == 5
    assert len(warning_messages) == 0
    assert any("num_mxfp8_blocks" in msg for msg in info_messages)


# ---------------------------------------------------------------------------
# Block-index dispatch (_parse_block_idx)
# ---------------------------------------------------------------------------


def test_parse_block_idx_valid():
    from vllm_omni.quantization.mixed_mxfp_config import _parse_block_idx

    assert _parse_block_idx("blocks.0.attn1.to_q") == 0
    assert _parse_block_idx("blocks.5.ffn.net.0.proj") == 5
    assert _parse_block_idx("blocks.40.norm1.weight") == 40


def test_parse_block_idx_non_block_prefixes():
    """Prefixes that do not start with 'blocks.N.' must return None."""
    from vllm_omni.quantization.mixed_mxfp_config import _parse_block_idx

    assert _parse_block_idx("condition_embedder.time_embedder.linear_1") is None
    assert _parse_block_idx("proj_out.weight") is None
    assert _parse_block_idx("model.layers.0.self_attn.q_proj") is None
    assert _parse_block_idx("scale_shift_table") is None


# ---------------------------------------------------------------------------
# SUPPORTED_QUANTIZATION_METHODS
# ---------------------------------------------------------------------------


def test_supported_methods_include_mxfp4_variants():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "mxfp4" in SUPPORTED_QUANTIZATION_METHODS
    assert "mxfp8" in SUPPORTED_QUANTIZATION_METHODS
    assert "mxfp8_mxfp4_dualscale" in SUPPORTED_QUANTIZATION_METHODS
