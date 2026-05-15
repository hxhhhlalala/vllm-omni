# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for merge_mxfp4_dualscale_checkpoint.py key-remapping helpers.

These are pure-Python unit tests that exercise the transformation functions
without loading any actual checkpoint files or requiring NPU hardware.
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# SUPPORTED_MODEL_TYPES
# ---------------------------------------------------------------------------


def test_supported_model_types_includes_a14b():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-T2V-A14B" in SUPPORTED_MODEL_TYPES
    assert "Wan2.2-I2V-A14B" in SUPPORTED_MODEL_TYPES


def test_supported_model_types_excludes_ti2v_5b():
    """TI2V-5B is explicitly NOT supported under W4A4 quantization."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-TI2V-5B" not in SUPPORTED_MODEL_TYPES


# ---------------------------------------------------------------------------
# _apply_rename_dict
# ---------------------------------------------------------------------------


def test_apply_rename_dict_self_attn_qkvo():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.0.self_attn.q.weight") == "blocks.0.attn1.to_q.weight"
    assert _apply_rename_dict("blocks.0.self_attn.k.weight") == "blocks.0.attn1.to_k.weight"
    assert _apply_rename_dict("blocks.0.self_attn.v.weight") == "blocks.0.attn1.to_v.weight"
    assert _apply_rename_dict("blocks.0.self_attn.o.weight") == "blocks.0.attn1.to_out.0.weight"


def test_apply_rename_dict_ffn():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.1.ffn.0.weight") == "blocks.1.ffn.net.0.proj.weight"
    assert _apply_rename_dict("blocks.1.ffn.2.weight") == "blocks.1.ffn.net.2.weight"


def test_apply_rename_dict_norm_order_swap():
    """norm2↔norm3 swap: quant tool uses norm1/norm3/norm2 order,
    Diffusers uses norm1/norm2/norm3."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.0.norm2.weight") == "blocks.0.norm3.weight"
    assert _apply_rename_dict("blocks.0.norm3.weight") == "blocks.0.norm2.weight"


def test_apply_rename_dict_cross_attn():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.0.cross_attn.q.weight") == "blocks.0.attn2.to_q.weight"
    assert _apply_rename_dict("blocks.0.cross_attn.k.weight") == "blocks.0.attn2.to_k.weight"
    assert _apply_rename_dict("blocks.0.cross_attn.v.weight") == "blocks.0.attn2.to_v.weight"
    assert _apply_rename_dict("blocks.0.cross_attn.o.weight") == "blocks.0.attn2.to_out.0.weight"


def test_apply_rename_dict_head_and_modulation():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("head.head.weight") == "proj_out.weight"


# ---------------------------------------------------------------------------
# _strip_mxfp4_wrapper
# ---------------------------------------------------------------------------


def test_strip_mxfp4_wrapper_linear_weight():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.weight") == "blocks.0.attn1.to_q.weight"


def test_strip_mxfp4_wrapper_linear_weight_scale():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.weight_scale") == "blocks.0.attn1.to_q.weight_scale"


def test_strip_mxfp4_wrapper_linear_weight_dual_scale():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert (
        _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.weight_dual_scale") == "blocks.0.attn1.to_q.weight_dual_scale"
    )


def test_strip_mxfp4_wrapper_linear_bias():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.bias") == "blocks.0.attn1.to_q.bias"


def test_strip_mxfp4_wrapper_div_mul_scale():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.div.mul_scale") == "blocks.0.attn1.to_q.mul_scale"


def test_strip_mxfp4_wrapper_noop_for_plain_weight():
    """MXFP8 / FLOAT tensors have no wrapper — must be returned unchanged."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.weight") == "blocks.0.attn1.to_q.weight"
    assert _strip_mxfp4_wrapper("blocks.0.norm_q.weight") == "blocks.0.norm_q.weight"
    assert _strip_mxfp4_wrapper("condition_embedder.time_embedder.linear_1.weight") == (
        "condition_embedder.time_embedder.linear_1.weight"
    )


# ---------------------------------------------------------------------------
# _classify_blocks
# ---------------------------------------------------------------------------


def test_classify_blocks_mixed_mxfp8_and_mxfp4():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _classify_blocks

    quant_meta = {
        "blocks.0.attn1.to_q.weight": "W8A8_MXFP8",
        "blocks.1.attn1.to_q.weight": "W8A8_MXFP8",
        "blocks.2.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "blocks.3.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "condition_embedder.time_embedder.linear_1.weight": "FLOAT",
    }
    block_types = _classify_blocks(quant_meta)
    assert block_types[0] == "mxfp8"
    assert block_types[1] == "mxfp8"
    assert block_types[2] == "mxfp4_dualscale"
    assert block_types[3] == "mxfp4_dualscale"
    # Non-block key must not produce an entry
    assert None not in block_types


def test_classify_blocks_float_entries_skipped():
    """FLOAT-typed tensors must not contribute to block classification."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _classify_blocks

    quant_meta = {
        "blocks.0.bias": "FLOAT",
        "blocks.1.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
    }
    block_types = _classify_blocks(quant_meta)
    assert 0 not in block_types
    assert block_types[1] == "mxfp4_dualscale"


def test_classify_blocks_empty():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _classify_blocks

    assert _classify_blocks({}) == {}


# ---------------------------------------------------------------------------
# _detect_num_mxfp8_blocks
# ---------------------------------------------------------------------------


def test_detect_num_mxfp8_blocks_with_mxfp8_present():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _detect_num_mxfp8_blocks

    quant_meta = {
        "blocks.0.attn1.to_q.weight": "W8A8_MXFP8",
        "blocks.1.attn1.to_q.weight": "W8A8_MXFP8",
        "blocks.2.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "blocks.3.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
    }
    assert _detect_num_mxfp8_blocks(quant_meta) == 2


def test_detect_num_mxfp8_blocks_without_mxfp8_keys():
    """When MXFP8 blocks are absent from quant_meta, the first MXFP4 block
    index equals num_mxfp8_blocks (msModelSlim may omit MXFP8 markers)."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _detect_num_mxfp8_blocks

    quant_meta = {
        "blocks.3.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "blocks.4.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
    }
    assert _detect_num_mxfp8_blocks(quant_meta) == 3


def test_detect_num_mxfp8_blocks_empty():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _detect_num_mxfp8_blocks

    assert _detect_num_mxfp8_blocks({}) == 0
