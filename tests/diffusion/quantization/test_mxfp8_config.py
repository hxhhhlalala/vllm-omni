# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for W8A8 MXFP8 quantization config."""

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


def test_mxfp8_config_creation():
    config = build_quant_config("mxfp8")
    assert config is not None
    assert config.get_name() == "mxfp8"


def test_mxfp8_config_default_offline():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config()
    assert config.is_checkpoint_mxfp8_serialized is False
    assert config.ignored_layers == []


def test_mxfp8_config_offline_mode():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    assert config.is_checkpoint_mxfp8_serialized is True


def test_mxfp8_config_ignored_layers():
    config = build_quant_config("mxfp8", ignored_layers=["proj_out"])
    assert "proj_out" in config.ignored_layers


def test_mxfp8_in_supported_methods():
    assert "mxfp8" in SUPPORTED_QUANTIZATION_METHODS


def test_mxfp8_from_config_offline():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config.from_config({"quant_method": "mxfp8"})
    assert config.is_checkpoint_mxfp8_serialized is True


def test_mxfp8_from_config_online():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config.from_config({"quant_method": "something_else"})
    assert config.is_checkpoint_mxfp8_serialized is False


def test_mxfp8_from_config_ignored_layers():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config.from_config({
        "quant_method": "mxfp8",
        "ignored_layers": ["proj_out"],
    })
    assert "proj_out" in config.ignored_layers


def test_mxfp8_from_config_modules_to_not_convert_fallback():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config.from_config({
        "quant_method": "mxfp8",
        "modules_to_not_convert": ["proj_out"],
    })
    assert "proj_out" in config.ignored_layers


# ---------------------------------------------------------------------------
# get_quant_method dispatch
# ---------------------------------------------------------------------------


def test_get_quant_method_non_linear_returns_none(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config()
    non_linear = mocker.Mock(spec=torch.nn.Module)
    assert config.get_quant_method(non_linear, "some.layer") is None


def test_get_quant_method_skipped_layer(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config(ignored_layers=["proj_out"])
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "proj_out")
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_quant_method_npu_online(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config, NPUMxfp8OnlineLinearMethod

    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False)
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp8OnlineLinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "blocks.0.attn.to_q")
    assert isinstance(method, NPUMxfp8OnlineLinearMethod)


def test_get_quant_method_npu_offline(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config, NPUMxfp8LinearMethod

    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp8LinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "blocks.0.attn.to_q")
    assert isinstance(method, NPUMxfp8LinearMethod)


def test_get_quant_method_non_npu_raises(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config()
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    with pytest.raises(NotImplementedError, match="NPU"):
        config.get_quant_method(layer, "blocks.0.attn.to_q")


# ---------------------------------------------------------------------------
# NPUMxfp8LinearMethod  (offline — mocked torch_npu)
# ---------------------------------------------------------------------------


class TestNPUMxfp8LinearMethod:
    FP8_DTYPE = object()  # sentinel; real dtype not needed in unit tests

    @pytest.fixture
    def quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_torch_npu(self, mocker):
        torch_npu = mocker.MagicMock()
        torch_npu.float8_e4m3fn = torch.int8  # stand-in dtype for shape checks
        torch_npu.float8_e8m0fnu = torch.uint8
        torch_npu.npu_dtype_cast = mocker.Mock(side_effect=lambda t, dt: t.to(torch.int8))
        torch_npu.npu_dynamic_mx_quant = mocker.Mock(
            return_value=(torch.zeros(4, 8, dtype=torch.int8), torch.zeros(4, 1, dtype=torch.uint8))
        )
        torch_npu.npu_quant_matmul = mocker.Mock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))
        mocker.patch("vllm_omni.quantization.mxfp8_config._torch_npu", torch_npu)
        mocker.patch("vllm_omni.quantization.mxfp8_config._get_torch_npu", return_value=torch_npu)
        return torch_npu

    def test_offline_process_weights_idempotent(self, quant_config, mock_torch_npu, mocker):
        from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod

        method = NPUMxfp8LinearMethod(quant_config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
        layer.weight_scale = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.uint8))
        layer._already_called_process_weights_after_loading = True

        mock_replace = mocker.patch("vllm_omni.quantization.mxfp8_config.replace_parameter")
        method.process_weights_after_loading(layer)
        # Should return early without calling replace_parameter
        mock_replace.assert_not_called()

    def test_apply_shape(self, quant_config, mock_torch_npu):
        from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod

        method = NPUMxfp8LinearMethod(quant_config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(4, 8, dtype=torch.int8))
        layer.weight_scale = torch.nn.Parameter(torch.zeros(1, 4, 2, dtype=torch.uint8))

        # Simulate quantize_activation → (x_q, x_scale)
        x = torch.randn(2, 3, 8)
        mock_torch_npu.npu_dynamic_mx_quant.return_value = (
            torch.zeros(6, 8, dtype=torch.int8),
            torch.zeros(6, 1, dtype=torch.uint8),
        )
        mock_torch_npu.npu_quant_matmul.return_value = torch.zeros(6, 4, dtype=torch.bfloat16)

        out = method.apply(layer, x)
        assert out.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# Integration: build_quant_config → OmniDiffusionConfig
# ---------------------------------------------------------------------------


def test_integration_mxfp8_string():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization_config="mxfp8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "mxfp8"


def test_integration_mxfp8_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "mxfp8", "is_checkpoint_mxfp8_serialized": True},
    )
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "mxfp8"
    assert config.quantization_config.is_checkpoint_mxfp8_serialized is True


def test_integration_dict_not_mutated():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original = {"method": "mxfp8", "is_checkpoint_mxfp8_serialized": True}
    copy = original.copy()
    OmniDiffusionConfig(model="test", quantization_config=original)
    assert original == copy


# ---------------------------------------------------------------------------
# NPU smoke tests (real hardware)
# ---------------------------------------------------------------------------


@npu_available
class TestNPUMxfp8Smoke:
    @pytest.fixture
    def quant_config(self):
        from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

        return DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False)

    @pytest.fixture
    def real_layer(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.bfloat16, device="npu"),
            requires_grad=False,
        )
        layer.logical_widths = [128]
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 128
        layer.orig_dtype = torch.bfloat16
        layer.weight_block_size = None
        return layer

    def test_npu_dynamic_mx_quant_shape(self, quant_config, real_layer):
        import torch_npu

        weight = real_layer.weight
        q, scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch_npu.float8_e4m3fn)
        assert q.shape == weight.shape
        assert q.dtype == torch_npu.float8_e4m3fn
        assert scale.shape[0] == weight.shape[0]

    def test_online_process_weights_after_loading(self, quant_config, real_layer):
        from vllm_omni.quantization.mxfp8_config import NPUMxfp8OnlineLinearMethod

        method = NPUMxfp8OnlineLinearMethod(quant_config)
        method.process_weights_after_loading(real_layer)

        # weight is (K, N) FP8, scale is (K_groups//2, N, 2)
        assert real_layer.weight.shape == (64, 128)
        assert hasattr(real_layer, "weight_scale")

    def test_online_apply_forward(self, quant_config):
        import torch_npu

        from vllm_omni.quantization.mxfp8_config import NPUMxfp8OnlineLinearMethod

        method = NPUMxfp8OnlineLinearMethod(quant_config)
        N, K = 128, 64
        weight_fp16 = torch.randn(N, K, dtype=torch.bfloat16, device="npu")
        w_q, scale_raw = torch_npu.npu_dynamic_mx_quant(weight_fp16, dst_type=torch_npu.float8_e4m3fn)
        K_groups = scale_raw.shape[1]
        if K_groups % 2 == 1:
            scale_raw = torch.cat([scale_raw, torch.zeros(N, 1, dtype=scale_raw.dtype, device="npu")], dim=1)
            K_groups += 1
        weight_scale = scale_raw.reshape(N, K_groups // 2, 2).transpose(0, 1).contiguous()
        w_q = w_q.transpose(0, 1).contiguous()

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        x = torch.randn(2, 16, K, dtype=torch.bfloat16, device="npu")
        output = method.apply(layer, x)
        assert output.shape == (2, 16, N)
        assert output.dtype == torch.bfloat16
