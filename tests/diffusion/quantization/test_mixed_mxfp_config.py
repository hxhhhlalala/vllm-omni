# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the mixed MXFP8 + MXFP4 DualScale quantization config."""

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

npu_available = pytest.mark.skipif(not current_omni_platform.is_npu(), reason="NPU platform not available.")


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def test_mixed_mxfp_config_creation():
    config = build_quant_config("mixed_mxfp", num_mxfp8_blocks=5)
    assert config is not None
    assert config.get_name() == "mixed_mxfp"


def test_mixed_mxfp_in_supported_methods():
    assert "mixed_mxfp" in SUPPORTED_QUANTIZATION_METHODS


def test_mixed_mxfp_config_defaults():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=3)
    assert config.num_mxfp8_blocks == 3
    assert config.is_checkpoint_serialized is False
    assert config.ignored_layers == []


def test_mixed_mxfp_config_serialized():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=5, is_checkpoint_serialized=True)
    assert config.is_checkpoint_serialized is True


def test_mixed_mxfp_from_config():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig.from_config({
        "quant_method": "mixed_mxfp",
        "num_mxfp8_blocks": 7,
        "is_checkpoint_serialized": True,
    })
    assert config.num_mxfp8_blocks == 7
    assert config.is_checkpoint_serialized is True


def test_mixed_mxfp_from_config_ignored_layers():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig.from_config({
        "quant_method": "mixed_mxfp",
        "num_mxfp8_blocks": 3,
        "ignored_layers": ["proj_out"],
    })
    assert "proj_out" in config.ignored_layers


def test_mixed_mxfp_from_config_modules_to_not_convert_fallback():
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig.from_config({
        "quant_method": "mixed_mxfp",
        "num_mxfp8_blocks": 3,
        "modules_to_not_convert": ["proj_out"],
    })
    assert "proj_out" in config.ignored_layers


# ---------------------------------------------------------------------------
# Block index parsing
# ---------------------------------------------------------------------------


def test_parse_block_idx_valid():
    from vllm_omni.quantization.mixed_mxfp_config import _parse_block_idx

    assert _parse_block_idx("blocks.0.attn1.to_q") == 0
    assert _parse_block_idx("blocks.5.ffn.net_0.proj") == 5
    assert _parse_block_idx("blocks.39.attn2.to_k") == 39


def test_parse_block_idx_no_match():
    from vllm_omni.quantization.mixed_mxfp_config import _parse_block_idx

    assert _parse_block_idx("condition_embedder.time_embedder.linear_1") is None
    assert _parse_block_idx("proj_out") is None
    assert _parse_block_idx("scale_shift_table") is None


# ---------------------------------------------------------------------------
# get_quant_method dispatch (mocked NPU)
# ---------------------------------------------------------------------------


def test_get_quant_method_non_linear_returns_none(mocker: MockerFixture):
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=5)
    non_linear = mocker.Mock(spec=torch.nn.Module)
    assert config.get_quant_method(non_linear, "some.layer") is None


def test_get_quant_method_skipped_layer(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=5, ignored_layers=["proj_out"])
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "proj_out")
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_quant_method_non_npu_raises(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=5)
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    with pytest.raises(NotImplementedError, match="NPU"):
        config.get_quant_method(layer, "blocks.0.attn1.to_q")


@pytest.mark.parametrize(
    "block_idx,num_mxfp8_blocks,expected_method",
    [
        # Within MXFP8 range → NPUMxfp8LinearMethod
        (0, 5, "NPUMxfp8LinearMethod"),
        (4, 5, "NPUMxfp8LinearMethod"),
        # At or beyond threshold → NPUMxfp4DualScaleLinearMethod
        (5, 5, "NPUMxfp4DualScaleLinearMethod"),
        (10, 5, "NPUMxfp4DualScaleLinearMethod"),
    ],
)
def test_get_quant_method_offline_dispatch(
    block_idx,
    num_mxfp8_blocks,
    expected_method,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig
    from vllm_omni.quantization.mxfp4_config import NPUMxfp4DualScaleLinearMethod
    from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod

    config = DiffusionMixedMXFPConfig(
        num_mxfp8_blocks=num_mxfp8_blocks,
        is_checkpoint_serialized=True,
    )
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp8LinearMethod, "__init__", lambda self, qc: None)
    mocker.patch.object(NPUMxfp4DualScaleLinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    prefix = f"blocks.{block_idx}.attn1.to_q"
    method = config.get_quant_method(layer, prefix)

    if expected_method == "NPUMxfp8LinearMethod":
        assert isinstance(method, NPUMxfp8LinearMethod)
    else:
        assert isinstance(method, NPUMxfp4DualScaleLinearMethod)


@pytest.mark.parametrize(
    "block_idx,num_mxfp8_blocks,expected_method",
    [
        (0, 5, "NPUMxfp8OnlineLinearMethod"),
        (4, 5, "NPUMxfp8OnlineLinearMethod"),
        (5, 5, "NPUMxfp4OnlineLinearMethod"),
        (10, 5, "NPUMxfp4OnlineLinearMethod"),
    ],
)
def test_get_quant_method_online_dispatch(
    block_idx,
    num_mxfp8_blocks,
    expected_method,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig
    from vllm_omni.quantization.mxfp4_config import NPUMxfp4OnlineLinearMethod
    from vllm_omni.quantization.mxfp8_config import NPUMxfp8OnlineLinearMethod

    config = DiffusionMixedMXFPConfig(
        num_mxfp8_blocks=num_mxfp8_blocks,
        is_checkpoint_serialized=False,
    )
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp8OnlineLinearMethod, "__init__", lambda self, qc: None)
    mocker.patch.object(NPUMxfp4OnlineLinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    prefix = f"blocks.{block_idx}.attn1.to_q"
    method = config.get_quant_method(layer, prefix)

    if expected_method == "NPUMxfp8OnlineLinearMethod":
        assert isinstance(method, NPUMxfp8OnlineLinearMethod)
    else:
        assert isinstance(method, NPUMxfp4OnlineLinearMethod)


def test_non_block_prefix_gets_mxfp4(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """Layers outside 'blocks.N.*' have no block index → fall through to MXFP4."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig
    from vllm_omni.quantization.mxfp4_config import NPUMxfp4DualScaleLinearMethod

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=5, is_checkpoint_serialized=True)
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp4DualScaleLinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    method = config.get_quant_method(layer, "proj_out")
    assert isinstance(method, NPUMxfp4DualScaleLinearMethod)


def test_num_mxfp8_blocks_zero(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """num_mxfp8_blocks=0 → all blocks use MXFP4."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig
    from vllm_omni.quantization.mxfp4_config import NPUMxfp4DualScaleLinearMethod

    config = DiffusionMixedMXFPConfig(num_mxfp8_blocks=0, is_checkpoint_serialized=True)
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp4DualScaleLinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    method = config.get_quant_method(layer, "blocks.0.attn1.to_q")
    assert isinstance(method, NPUMxfp4DualScaleLinearMethod)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_integration_mixed_mxfp_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "method": "mixed_mxfp",
            "num_mxfp8_blocks": 5,
            "is_checkpoint_serialized": True,
        },
    )
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "mixed_mxfp"
    assert config.quantization_config.num_mxfp8_blocks == 5
    assert config.quantization_config.is_checkpoint_serialized is True


def test_integration_dict_not_mutated():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original = {"method": "mixed_mxfp", "num_mxfp8_blocks": 5, "is_checkpoint_serialized": True}
    copy = original.copy()
    OmniDiffusionConfig(model="test", quantization_config=original)
    assert original == copy


def test_integration_from_config_json_format():
    """Verify the exact dict that merge_mxfp4_dualscale_checkpoint.py injects into config.json."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMixedMXFPConfig

    config_json_entry = {
        "quant_method": "mixed_mxfp",
        "num_mxfp8_blocks": 5,
        "is_checkpoint_serialized": True,
    }
    config = DiffusionMixedMXFPConfig.from_config(config_json_entry)
    assert config.num_mxfp8_blocks == 5
    assert config.is_checkpoint_serialized is True
