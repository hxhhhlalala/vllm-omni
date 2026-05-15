# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for transformer quant-config auto-detection and cascade propagation.

The loader path at pipeline_wan2_2.py carries two quantization contracts:

  create_transformer_from_config (~L137)
    - Reads quantization_config from config.json (injected by the merge scripts)
    - Auto-detects the quant method when no CLI quant_config is provided
    - Rejects method mismatches (CLI vs disk)
    - Upgrades online → offline when disk marks is_checkpoint_*_serialized=True
    - Rebuilds when the active num_mxfp8_blocks differs from the disk value

  Wan22Pipeline._create_transformer (~L456)
    - Propagates the auto-detected config to od_config so the second transformer
      in a cascade model reuses the same config rather than re-reading independently
    - Does NOT overwrite od_config.quantization_config when it is already set

All tests are pure-CPU and do not load model weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import create_transformer_from_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Minimum config that create_transformer_from_config accepts without raising.
_MIN_CFG: dict = {"patch_size": [1, 2, 2]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_transformer():
    """Return (FakeTransformer class, captured list).

    Each _FakeTransformer.__init__ call appends its **kwargs to captured,
    letting tests inspect what quant_config was passed per transformer.
    """
    captured: list[dict] = []

    class _FakeTransformer:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    return _FakeTransformer, captured


class _FakePipeline:
    """Minimal stand-in exposing only what _create_transformer needs from self."""

    def __init__(self, od_config: SimpleNamespace) -> None:
        self.od_config = od_config

    # Bind the real unbound method so the tests exercise production code.
    _create_transformer = wan22_module.Wan22Pipeline._create_transformer


# ---------------------------------------------------------------------------
# create_transformer_from_config — auto-detection
# ---------------------------------------------------------------------------


def test_create_transformer_detects_mxfp8_serialized_from_config_json(monkeypatch):
    """When config.json carries MXFP8 quant and no CLI quant_config is provided,
    the transformer must receive a DiffusionMXFP8Config with serialized=True."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    create_transformer_from_config(config)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8Config)
    assert qc.is_checkpoint_mxfp8_serialized is True


def test_create_transformer_detects_mxfp4_dualscale_from_config_json(monkeypatch):
    """config.json with mxfp8_mxfp4_dualscale + num_mxfp8_blocks must produce
    a DiffusionMXFP8MXFP4DualScaleConfig with the correct block count and
    serialized flag."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 7,
            "is_checkpoint_serialized": True,
        },
    }
    create_transformer_from_config(config)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8MXFP4DualScaleConfig)
    assert qc.num_mxfp8_blocks == 7
    assert qc.is_checkpoint_serialized is True


def test_create_transformer_without_quantization_config_passes_no_quant(monkeypatch):
    """A plain BF16 config.json (no quantization_config key) must result in no
    quant_config being passed to WanTransformer3DModel."""
    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    create_transformer_from_config(_MIN_CFG)

    assert "quant_config" not in captured[0]


# ---------------------------------------------------------------------------
# create_transformer_from_config — method-mismatch guard
# ---------------------------------------------------------------------------


def test_create_transformer_rejects_method_mismatch(monkeypatch):
    """Passing a CLI quant_config whose get_name() differs from the config.json
    quant_method must raise ValueError immediately (prevents silent weight corruption)."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    FakeTransformer, _ = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    fp8_cli = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic")
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    with pytest.raises(ValueError, match="quant_method"):
        create_transformer_from_config(config, quant_config=fp8_cli)


# ---------------------------------------------------------------------------
# create_transformer_from_config — online → offline upgrade
# ---------------------------------------------------------------------------


def test_create_transformer_upgrades_to_serialized_when_disk_marks_it(monkeypatch):
    """CLI passes online (is_checkpoint_mxfp8_serialized=False) but config.json
    marks is_checkpoint_mxfp8_serialized=True → must switch to offline (serialized)
    so that pre-quantized FP8 tensors are loaded correctly."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    online_cli = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False)
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    create_transformer_from_config(config, quant_config=online_cli)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8Config)
    assert qc.is_checkpoint_mxfp8_serialized is True


# ---------------------------------------------------------------------------
# create_transformer_from_config — num_mxfp8_blocks rebuild
# ---------------------------------------------------------------------------


def test_create_transformer_rebuilds_when_num_mxfp8_blocks_differs(monkeypatch):
    """When the active quant_config has num_mxfp8_blocks=5 but config.json says 10,
    the config must be rebuilt from disk so block routing is authoritative for
    this specific transformer."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    stale = DiffusionMXFP8MXFP4DualScaleConfig(num_mxfp8_blocks=5, is_checkpoint_serialized=True)
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 10,
            "is_checkpoint_serialized": True,
        },
    }
    create_transformer_from_config(config, quant_config=stale)

    qc = captured[0].get("quant_config")
    assert isinstance(qc, DiffusionMXFP8MXFP4DualScaleConfig)
    assert qc.num_mxfp8_blocks == 10


def test_create_transformer_does_not_rebuild_when_num_mxfp8_blocks_matches(monkeypatch):
    """When the active quant_config already has the correct num_mxfp8_blocks,
    the same instance must be passed through unchanged (no unnecessary rebuild)."""
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    matching = DiffusionMXFP8MXFP4DualScaleConfig(num_mxfp8_blocks=5, is_checkpoint_serialized=True)
    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 5,
            "is_checkpoint_serialized": True,
        },
    }
    create_transformer_from_config(config, quant_config=matching)

    assert captured[0].get("quant_config") is matching


# ---------------------------------------------------------------------------
# Wan22Pipeline._create_transformer — od_config propagation
# ---------------------------------------------------------------------------


def test_pipeline_create_transformer_propagates_quant_config_to_od_config(monkeypatch):
    """When od_config.quantization_config is None, _create_transformer must
    auto-detect the quant method from config.json and propagate the built config
    back to od_config so the next call can reuse it."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, _ = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    od_config = SimpleNamespace(quantization_config=None)
    pipeline = _FakePipeline(od_config)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    pipeline._create_transformer(config)

    assert isinstance(od_config.quantization_config, DiffusionMXFP8Config)
    assert od_config.quantization_config.is_checkpoint_mxfp8_serialized is True


def test_pipeline_create_transformer_does_not_overwrite_existing_od_config(monkeypatch):
    """If od_config.quantization_config is already set (propagated from the first
    transformer), _create_transformer must leave it unchanged — the propagated
    config is the authority for the cascade."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, _ = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    existing = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    od_config = SimpleNamespace(quantization_config=existing)
    pipeline = _FakePipeline(od_config)

    config = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
        },
    }
    pipeline._create_transformer(config)

    assert od_config.quantization_config is existing


# ---------------------------------------------------------------------------
# Wan22Pipeline._create_transformer — cascade contracts
# ---------------------------------------------------------------------------


def test_pipeline_cascade_both_transformers_get_mxfp8_serialized_config(monkeypatch):
    """Cascade model (transformer + transformer_2) with MXFP8 checkpoint:
    - First transformer:  auto-detects serialized config, propagates to od_config.
    - Second transformer: reuses the propagated config (same instance).
    Both must receive is_checkpoint_mxfp8_serialized=True."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    od_config = SimpleNamespace(quantization_config=None)
    pipeline = _FakePipeline(od_config)

    mxfp8_qc = {"quant_method": "mxfp8", "is_checkpoint_mxfp8_serialized": True}
    pipeline._create_transformer({**_MIN_CFG, "quantization_config": mxfp8_qc})
    pipeline._create_transformer({**_MIN_CFG, "quantization_config": mxfp8_qc})

    assert len(captured) == 2
    for i, kwargs in enumerate(captured):
        qc = kwargs.get("quant_config")
        assert isinstance(qc, DiffusionMXFP8Config), f"transformer[{i}]: expected DiffusionMXFP8Config, got {type(qc)}"
        assert qc.is_checkpoint_mxfp8_serialized is True, f"transformer[{i}]: expected serialized=True"

    # Second transformer must reuse the propagated instance — no unnecessary rebuild.
    assert captured[0]["quant_config"] is captured[1]["quant_config"]


def test_pipeline_cascade_mxfp4_dualscale_each_transformer_gets_correct_num_blocks(monkeypatch):
    """Cascade with mxfp8_mxfp4_dualscale where transformer and transformer_2 have
    different num_mxfp8_blocks in their config.json.

    Expected outcome:
      transformer   → num_mxfp8_blocks=5  (auto-detected, propagated to od_config)
      transformer_2 → num_mxfp8_blocks=10 (rebuilt from disk because 10 ≠ 5)
      od_config     → num_mxfp8_blocks=5  (unchanged; transformer_2's rebuild is local)
    """
    from vllm_omni.quantization.mixed_mxfp_config import DiffusionMXFP8MXFP4DualScaleConfig

    FakeTransformer, captured = _make_fake_transformer()
    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    od_config = SimpleNamespace(quantization_config=None)
    pipeline = _FakePipeline(od_config)

    cfg1 = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 5,
            "is_checkpoint_serialized": True,
        },
    }
    cfg2 = {
        **_MIN_CFG,
        "quantization_config": {
            "quant_method": "mxfp8_mxfp4_dualscale",
            "num_mxfp8_blocks": 10,
            "is_checkpoint_serialized": True,
        },
    }

    pipeline._create_transformer(cfg1)
    pipeline._create_transformer(cfg2)

    assert len(captured) == 2
    qc1 = captured[0].get("quant_config")
    qc2 = captured[1].get("quant_config")

    assert isinstance(qc1, DiffusionMXFP8MXFP4DualScaleConfig)
    assert isinstance(qc2, DiffusionMXFP8MXFP4DualScaleConfig)
    assert qc1.num_mxfp8_blocks == 5, f"transformer expected 5 blocks, got {qc1.num_mxfp8_blocks}"
    assert qc2.num_mxfp8_blocks == 10, f"transformer_2 expected 10 blocks, got {qc2.num_mxfp8_blocks}"

    # od_config retains the first transformer's config; the rebuild was local.
    assert od_config.quantization_config.num_mxfp8_blocks == 5
