# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for W4A4 MXFP4 quantization config (single-scale and dual-scale)."""

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


def test_mxfp4_config_creation():
    config = build_quant_config("mxfp4")
    assert config is not None
    assert config.get_name() == "mxfp4"


def test_mxfp4_config_default_online():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config()
    assert config.is_checkpoint_mxfp4_serialized is False
    assert config.ignored_layers == []


def test_mxfp4_config_offline_mode():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)
    assert config.is_checkpoint_mxfp4_serialized is True


def test_mxfp4_config_ignored_layers():
    config = build_quant_config("mxfp4", ignored_layers=["proj_out"])
    assert "proj_out" in config.ignored_layers


def test_mxfp4_in_supported_methods():
    assert "mxfp4" in SUPPORTED_QUANTIZATION_METHODS


def test_mxfp4_from_config_offline():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config.from_config({"quant_method": "mxfp4"})
    assert config.is_checkpoint_mxfp4_serialized is True


def test_mxfp4_from_config_online():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config.from_config({"quant_method": "bf16"})
    assert config.is_checkpoint_mxfp4_serialized is False


def test_mxfp4_from_config_modules_to_not_convert_fallback():
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config.from_config({
        "quant_method": "mxfp4",
        "modules_to_not_convert": ["proj_out"],
    })
    assert "proj_out" in config.ignored_layers


# ---------------------------------------------------------------------------
# get_quant_method dispatch
# ---------------------------------------------------------------------------


def test_get_quant_method_non_linear_returns_none(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config()
    non_linear = mocker.Mock(spec=torch.nn.Module)
    assert config.get_quant_method(non_linear, "some.layer") is None


def test_get_quant_method_skipped_layer(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config(ignored_layers=["proj_out"])
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "proj_out")
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_quant_method_npu_online(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4OnlineLinearMethod

    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=False)
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp4OnlineLinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "blocks.0.ffn.net_0.proj")
    assert isinstance(method, NPUMxfp4OnlineLinearMethod)


def test_get_quant_method_npu_offline(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4LinearMethod

    config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)
    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(NPUMxfp4LinearMethod, "__init__", lambda self, qc: None)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, "blocks.0.ffn.net_0.proj")
    assert isinstance(method, NPUMxfp4LinearMethod)


def test_get_quant_method_non_npu_raises(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

    config = DiffusionMXFP4Config()
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    with pytest.raises(NotImplementedError, match="NPU"):
        config.get_quant_method(layer, "blocks.0.ffn.net_0.proj")


# ---------------------------------------------------------------------------
# NPUMxfp4LinearMethod  (offline, mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_torch_npu_fp4(mocker):
    torch_npu = mocker.MagicMock()
    torch_npu.float4_e2m1fn_x2 = torch.int8  # stand-in for shape-only checks
    torch_npu.float8_e8m0fnu = torch.uint8
    torch_npu.npu_dtype_cast = mocker.Mock(side_effect=lambda t, dt: t.to(torch.int8))
    torch_npu.npu_dynamic_mx_quant = mocker.Mock(
        return_value=(torch.zeros(4, 8, dtype=torch.int8), torch.zeros(4, 1, dtype=torch.uint8))
    )
    torch_npu.npu_quant_matmul = mocker.Mock(return_value=torch.zeros(6, 4, dtype=torch.bfloat16))
    mocker.patch("vllm_omni.quantization.mxfp4_config._get_torch_npu", return_value=torch_npu)
    mocker.patch("vllm_omni.quantization.mxfp8_config._get_torch_npu", return_value=torch_npu)
    mocker.patch("vllm_omni.quantization.mxfp8_config._torch_npu", torch_npu)
    return torch_npu


class TestNPUMxfp4OfflineLinearMethod:
    def test_process_weights_idempotent(self, mock_torch_npu_fp4, mocker):
        from vllm_omni.quantization.mxfp4_config import NPUMxfp4LinearMethod

        config = mocker.Mock()
        method = NPUMxfp4LinearMethod(config)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
        layer.weight_scale = torch.nn.Parameter(torch.zeros(8, 1, dtype=torch.uint8))
        layer._already_called_process_weights_after_loading = True

        mock_replace = mocker.patch("vllm_omni.quantization.mxfp4_config.replace_parameter")
        method.process_weights_after_loading(layer)
        mock_replace.assert_not_called()

    def test_apply_shape(self, mock_torch_npu_fp4, mocker):
        from vllm_omni.quantization.mxfp4_config import NPUMxfp4LinearMethod

        config = mocker.Mock()
        method = NPUMxfp4LinearMethod(config)

        N, K = 4, 8
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(N, K, dtype=torch.int8))
        layer.weight_scale = torch.nn.Parameter(torch.zeros(N, 1, 2, dtype=torch.uint8))

        mock_torch_npu_fp4.npu_dynamic_mx_quant.return_value = (
            torch.zeros(6, K, dtype=torch.int8),
            torch.zeros(6, 1, dtype=torch.uint8),
        )
        mock_torch_npu_fp4.npu_quant_matmul.return_value = torch.zeros(6, N, dtype=torch.bfloat16)

        x = torch.randn(2, 3, K)
        out = method.apply(layer, x)
        assert out.shape == (2, 3, N)


# ---------------------------------------------------------------------------
# NPUMxfp4DualScaleLinearMethod  (mocked)
# ---------------------------------------------------------------------------


class TestNPUMxfp4DualScaleLinearMethod:
    def test_apply_shape_and_dual_scale(self, mock_torch_npu_fp4, mocker):
        from vllm_omni.quantization.mxfp4_config import NPUMxfp4DualScaleLinearMethod

        config = mocker.Mock()
        method = NPUMxfp4DualScaleLinearMethod(config)

        N, K = 4, 8
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(N, K, dtype=torch.int8))
        layer.weight_scale = torch.nn.Parameter(torch.zeros(N, 1, 2, dtype=torch.uint8))
        layer.weight_dual_scale = torch.nn.Parameter(torch.ones(N, dtype=torch.float32))
        layer.mul_scale = torch.nn.Parameter(torch.ones(K, dtype=torch.float32))

        mock_torch_npu_fp4.npu_dynamic_mx_quant.return_value = (
            torch.zeros(6, K, dtype=torch.int8),
            torch.zeros(6, 1, dtype=torch.uint8),
        )
        mock_torch_npu_fp4.npu_quant_matmul.return_value = torch.zeros(6, N, dtype=torch.bfloat16)

        x = torch.randn(2, 3, K)
        out = method.apply(layer, x)
        assert out.shape == (2, 3, N)

    def test_mul_scale_applied_before_quant(self, mock_torch_npu_fp4, mocker):
        """Verify mul_scale pre-multiplied before npu_dynamic_mx_quant call."""
        from vllm_omni.quantization.mxfp4_config import NPUMxfp4DualScaleLinearMethod

        config = mocker.Mock()
        method = NPUMxfp4DualScaleLinearMethod(config)

        N, K = 4, 8
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(N, K, dtype=torch.int8))
        layer.weight_scale = torch.nn.Parameter(torch.zeros(N, 1, 2, dtype=torch.uint8))
        # Dual scale of 2 → output should be scaled by 2
        layer.weight_dual_scale = torch.nn.Parameter(torch.full((N,), 2.0, dtype=torch.float32))
        layer.mul_scale = torch.nn.Parameter(torch.ones(K, dtype=torch.float32))

        # quant matmul returns ones; after dual_scale multiply → twos
        mock_torch_npu_fp4.npu_dynamic_mx_quant.return_value = (
            torch.zeros(6, K, dtype=torch.int8),
            torch.zeros(6, 1, dtype=torch.uint8),
        )
        mock_torch_npu_fp4.npu_quant_matmul.return_value = torch.ones(6, N, dtype=torch.bfloat16)

        x = torch.randn(2, 3, K)
        out = method.apply(layer, x)
        assert torch.allclose(out.float(), torch.full((2, 3, N), 2.0))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_integration_mxfp4_string():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization_config="mxfp4")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "mxfp4"


def test_integration_mxfp4_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "mxfp4", "is_checkpoint_mxfp4_serialized": True},
    )
    assert config.quantization_config.get_name() == "mxfp4"
    assert config.quantization_config.is_checkpoint_mxfp4_serialized is True


def test_integration_dict_not_mutated():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original = {"method": "mxfp4", "is_checkpoint_mxfp4_serialized": True}
    copy = original.copy()
    OmniDiffusionConfig(model="test", quantization_config=original)
    assert original == copy


# ---------------------------------------------------------------------------
# NPU smoke tests (real hardware)
# ---------------------------------------------------------------------------


@npu_available
class TestNPUMxfp4Smoke:
    @pytest.fixture
    def quant_config(self):
        from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

        return DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=False)

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

    def test_npu_dynamic_mx_quant_fp4_shape(self, quant_config, real_layer):
        import torch_npu

        weight = real_layer.weight
        q, scale = torch_npu.npu_dynamic_mx_quant(weight)
        assert q.dtype == torch_npu.float4_e2m1fn_x2
        assert scale.shape[0] == weight.shape[0]

    def test_online_process_weights_after_loading(self, quant_config, real_layer):
        from vllm_omni.quantization.mxfp4_config import NPUMxfp4OnlineLinearMethod

        method = NPUMxfp4OnlineLinearMethod(quant_config)
        method.process_weights_after_loading(real_layer)

        # weight stays (N, K) FP4; scale is (N, S//2, 2)
        assert real_layer.weight.shape == (128, 64)
        assert hasattr(real_layer, "weight_scale")
        assert real_layer.weight_scale.ndim == 3

    def test_online_apply_forward(self, quant_config):
        import torch_npu

        from vllm_omni.quantization.mxfp4_config import NPUMxfp4OnlineLinearMethod

        method = NPUMxfp4OnlineLinearMethod(quant_config)
        N, K = 128, 64

        weight_fp16 = torch.randn(N, K, dtype=torch.bfloat16, device="npu")
        w_q, scale_raw = torch_npu.npu_dynamic_mx_quant(weight_fp16)
        weight_scale = scale_raw.reshape(N, -1, 2).contiguous()

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        x = torch.randn(2, 16, K, dtype=torch.bfloat16, device="npu")
        output = method.apply(layer, x)
        assert output.shape == (2, 16, N)
        assert output.dtype == torch.bfloat16

    def test_dual_scale_apply_forward(self):
        import torch_npu

        from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4DualScaleLinearMethod

        config = DiffusionMXFP4Config(is_checkpoint_mxfp4_serialized=True)
        method = NPUMxfp4DualScaleLinearMethod(config)
        N, K = 128, 64

        weight_fp16 = torch.randn(N, K, dtype=torch.bfloat16, device="npu")
        w_q, scale_raw = torch_npu.npu_dynamic_mx_quant(weight_fp16)
        S = scale_raw.shape[1]
        if S % 2 == 1:
            scale_raw = torch.cat([scale_raw, torch.zeros(N, 1, dtype=scale_raw.dtype, device="npu")], dim=1)
            S += 1
        weight_scale = scale_raw.reshape(N, S // 2, 2).contiguous()

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.weight_dual_scale = torch.nn.Parameter(torch.ones(N, dtype=torch.float32, device="npu"))
        layer.mul_scale = torch.nn.Parameter(torch.ones(K, dtype=torch.float32, device="npu"))

        x = torch.randn(2, 16, K, dtype=torch.bfloat16, device="npu")
        output = method.apply(layer, x)
        assert output.shape == (2, 16, N)
        assert output.dtype == torch.bfloat16
