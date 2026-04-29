#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge mixed MXFP8 + W4A4_MXFP4_DUALSCALE quantized Wan2.2 weights into HF Diffusers format.

The quantization tool (msmodelslim) produces a mixed-precision checkpoint where:
  - Early blocks (0..num_mxfp8_blocks-1) use W8A8_MXFP8
  - Remaining blocks use W4A4_MXFP4_DUALSCALE

W4A4_MXFP4_DUALSCALE wraps each linear in two sub-modules:
  X.linear.weight / weight_scale / weight_dual_scale / bias  – quantized weight tensors
  X.div.mul_scale                                            – per-input-channel activation scale

After merging, the vllm-omni runtime uses DiffusionMixedMXFPConfig (quant_method="mixed_mxfp")
which dispatches NPUMxfp8LinearMethod for the MXFP8 blocks and NPUMxfp4DualScaleLinearMethod
for the MXFP4 blocks.

Usage:
  python merge_mxfp4_dualscale_checkpoint.py \\
      --model-type      Wan2.2-T2V-A14B \\
      --original-model  /path/to/Wan2.2-T2V-A14B-Diffusers \\
      --quant-path      /path/to/msmodelslim-output \\
      --output-path     /path/to/merged-output \\
      --num-mxfp8-blocks 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import warnings
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Shared key rename dict  (quant-tool naming → diffusers naming)
# Copied from merge_mxfp8_checkpoint.py; applied to both MXFP8 and MXFP4 blocks.
# ---------------------------------------------------------------------------

TRANSFORMER_KEYS_RENAME_DICT: dict[str, str] = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Norm order swap
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # Self-attention
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    # Cross-attention
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
    # I2V image embedder
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
}

SUPPORTED_MODEL_TYPES = ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B", "Wan2.2-TI2V-5B"]
CASCADE_MODEL_TYPES = {"Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"}

# Self-attention projections that are fused into to_qkv
_SELF_ATTN_QKV = {"attn1.to_q", "attn1.to_k", "attn1.to_v"}


# ---------------------------------------------------------------------------
# Key classification helpers
# ---------------------------------------------------------------------------

def _apply_rename_dict(key: str) -> str:
    for src, dst in TRANSFORMER_KEYS_RENAME_DICT.items():
        key = key.replace(src, dst)
    return key


def _strip_linear_wrapper(key: str) -> tuple[str, str | None]:
    """Strip .linear. or .div. wrappers from MXFP4 keys.

    Returns (stripped_key, attr) where attr is one of:
        'weight', 'weight_scale', 'weight_dual_scale', 'bias', 'mul_scale', None
    """
    for attr in ("weight_dual_scale", "weight_scale", "weight", "bias"):
        suffix = f".linear.{attr}"
        if key.endswith(suffix):
            return key[: -len(suffix)], attr

    if key.endswith(".div.mul_scale"):
        return key[: -len(".div.mul_scale")], "mul_scale"

    return key, None


# ---------------------------------------------------------------------------
# Block-level type detection
# ---------------------------------------------------------------------------

def _detect_block_types(quant_meta: dict[str, str]) -> dict[int, str]:
    """Return {block_idx: 'mxfp8' | 'mxfp4'} from quant_meta."""
    block_types: dict[int, str] = {}
    for key, qtype in quant_meta.items():
        parts = key.split(".")
        if len(parts) < 2 or parts[0] != "blocks" or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        if idx in block_types:
            continue
        if qtype.startswith("W4A4_MXFP4"):
            block_types[idx] = "mxfp4"
        elif qtype.startswith("W8A8_MXFP8"):
            block_types[idx] = "mxfp8"
    return block_types


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _remap_mxfp8_key(key: str) -> str:
    """Apply rename dict to an MXFP8-format key (no wrappers)."""
    return _apply_rename_dict(key)


def _remap_mxfp4_keys(
    state_dict: dict[str, torch.Tensor],
    quant_meta: dict[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Remap MXFP4 DUALSCALE keys from an MXFP4 block's state dict:
      1. Apply rename dict to map quant-tool names to diffusers names.
      2. Strip .linear. and .div. wrappers (present only on quantized tensors).
      3. Pre-fuse mul_scale for self-attention to_qkv (use q's value; k/v are identical).

    FLOAT tensors (biases, norms, etc.) have no wrappers; only the rename dict is applied.

    Returns (new_state_dict, new_quant_meta).
    """
    new_state: dict[str, torch.Tensor] = {}
    new_meta: dict[str, str] = {}

    # First pass: strip wrappers for all tensors in this block
    for key, tensor in state_dict.items():
        renamed = _apply_rename_dict(key)
        base, attr = _strip_linear_wrapper(renamed)

        if attr is None:
            # FLOAT tensor (norm weight, bias without wrapper, q_rot, etc.) — rename only
            new_state[renamed] = tensor
            if key in quant_meta:
                new_meta[renamed] = quant_meta[key]
        else:
            out_key = f"{base}.{attr}"
            new_state[out_key] = tensor
            if key in quant_meta:
                new_meta[out_key] = quant_meta[key]

    # Second pass: pre-fuse mul_scale for self-attention QKV
    # (attn1.to_q/k/v share the same input → same mul_scale; keep to_q's value)
    fused: dict[str, torch.Tensor] = {}
    fused_meta: dict[str, str] = {}
    to_drop: set[str] = set()

    for key in list(new_state.keys()):
        if not key.endswith(".mul_scale"):
            continue
        # Derive projection path: "blocks.N.attn1.to_q.mul_scale"
        base = key[: -len(".mul_scale")]  # "blocks.N.attn1.to_q"

        # Check if this projection is one of Q/K/V in self-attention
        matched_qkv = None
        for qkv in _SELF_ATTN_QKV:
            if base.endswith(qkv):
                matched_qkv = qkv
                break

        if matched_qkv is None:
            continue  # cross-attn or ffn: keep as individual mul_scale

        # Determine the fused key
        prefix_end = base[: -len(matched_qkv)]  # "blocks.N."
        fused_key = f"{prefix_end}attn1.to_qkv.mul_scale"

        if matched_qkv == "attn1.to_q":
            # Use Q's mul_scale as the representative
            fused[fused_key] = new_state[key]
            fused_meta[fused_key] = new_meta.get(key, "")
        else:
            # Verify K/V mul_scale matches Q (warn if not)
            if fused_key in fused:
                q_scale = fused[fused_key]
                kv_scale = new_state[key]
                if not torch.allclose(q_scale.float(), kv_scale.float(), atol=1e-5):
                    warnings.warn(
                        f"Q/K/V mul_scale differ for {prefix_end}attn1 (max_delta="
                        f"{(q_scale.float() - kv_scale.float()).abs().max().item():.6f}). "
                        "Using Q's mul_scale for fused to_qkv."
                    )
        to_drop.add(key)

    for key in to_drop:
        new_state.pop(key, None)
        new_meta.pop(key, None)
    new_state.update(fused)
    new_meta.update(fused_meta)

    return new_state, new_meta


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_safetensors(directory: pathlib.Path) -> dict[str, torch.Tensor]:
    candidates = sorted(directory.glob("quant_model_weight*.safetensors"))
    if not candidates:
        candidates = sorted(directory.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    state_dict: dict[str, torch.Tensor] = {}
    for f in candidates:
        state_dict.update(load_file(str(f)))
    return state_dict


def _load_quant_meta(directory: pathlib.Path) -> dict[str, str]:
    candidates = sorted(directory.glob("quant_model_description*.json"))
    if not candidates:
        print(f"  WARNING: No quant_model_description*.json in {directory}; treating all tensors as FLOAT.")
        return {}
    with open(candidates[0]) as f:
        return json.load(f)


def _get_transformer_dirs(model_type: str) -> list[str]:
    return ["transformer", "transformer_2"] if model_type in CASCADE_MODEL_TYPES else ["transformer"]


def _get_quant_subdir(model_type: str, quant_path: pathlib.Path, transformer_dir: str) -> pathlib.Path:
    if model_type in CASCADE_MODEL_TYPES:
        sub = "high_noise_model" if transformer_dir == "transformer" else "low_noise_model"
        return quant_path / sub
    return quant_path


# ---------------------------------------------------------------------------
# Per-transformer conversion
# ---------------------------------------------------------------------------

def _convert_transformer(
    model_type: str,
    quant_subdir: pathlib.Path,
    output_dir: pathlib.Path,
    original_transformer_dir: pathlib.Path,
    num_mxfp8_blocks: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading quantized weights from {quant_subdir} …")
    state_dict = _load_safetensors(quant_subdir)
    quant_meta = _load_quant_meta(quant_subdir)
    print(f"  {len(state_dict)} tensors, {len(quant_meta)} quant_meta entries")

    # Detect block types from quant_meta
    block_types = _detect_block_types(quant_meta)
    auto_mxfp8_count = sum(1 for t in block_types.values() if t == "mxfp8")
    auto_mxfp4_count = sum(1 for t in block_types.values() if t == "mxfp4")
    print(f"  Detected: {auto_mxfp8_count} MXFP8 blocks, {auto_mxfp4_count} MXFP4 blocks")

    # Resolve num_mxfp8_blocks
    if num_mxfp8_blocks is None:
        num_mxfp8_blocks = auto_mxfp8_count
        print(f"  Auto-detected num_mxfp8_blocks={num_mxfp8_blocks}")
    else:
        if num_mxfp8_blocks != auto_mxfp8_count:
            warnings.warn(
                f"--num-mxfp8-blocks={num_mxfp8_blocks} but auto-detected {auto_mxfp8_count}; "
                "using the provided value."
            )

    # Split state_dict into MXFP8 and MXFP4 parts
    mxfp8_state: dict[str, torch.Tensor] = {}
    mxfp8_meta: dict[str, str] = {}
    mxfp4_state: dict[str, torch.Tensor] = {}
    mxfp4_meta: dict[str, str] = {}
    other_state: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        qtype = quant_meta.get(key, "FLOAT")
        parts = key.split(".")
        is_block = len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()

        if is_block:
            idx = int(parts[1])
            if block_types.get(idx) == "mxfp4":
                mxfp4_state[key] = tensor
                mxfp4_meta[key] = qtype
            else:
                mxfp8_state[key] = tensor
                mxfp8_meta[key] = qtype
        else:
            other_state[key] = tensor

    # Remap MXFP8 block keys (standard rename dict)
    out_state: dict[str, torch.Tensor] = {}
    out_meta: dict[str, str] = {}

    for key, tensor in mxfp8_state.items():
        new_key = _remap_mxfp8_key(key)
        out_state[new_key] = tensor
        if key in mxfp8_meta:
            out_meta[new_key] = mxfp8_meta[key]

    # Remap MXFP4 block keys (strip wrappers + pre-fuse mul_scale)
    fp4_state, fp4_meta = _remap_mxfp4_keys(mxfp4_state, mxfp4_meta)
    out_state.update(fp4_state)
    out_meta.update(fp4_meta)

    # Remap non-block keys (embeddings, norms, etc.)
    for key, tensor in other_state.items():
        new_key = _remap_mxfp8_key(key)
        out_state[new_key] = tensor

    # Save
    out_weights = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(out_state, str(out_weights))
    print(f"  Saved {len(out_state)} tensors → {out_weights}")

    out_meta_path = output_dir / "quant_model_description.json"
    with open(out_meta_path, "w") as f:
        json.dump(out_meta, f, indent=2)

    # Inject quantization_config into config.json
    src_config = original_transformer_dir / "config.json"
    if src_config.is_file():
        with open(src_config) as f:
            config = json.load(f)
        config["quantization_config"] = {
            "quant_method": "mixed_mxfp",
            "num_mxfp8_blocks": num_mxfp8_blocks,
            "is_checkpoint_serialized": True,
        }
        out_config = output_dir / "config.json"
        with open(out_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Injected quantization_config (mixed_mxfp, num_mxfp8_blocks={num_mxfp8_blocks}) → {out_config}")
    else:
        print(f"  WARNING: No config.json found at {src_config}")


# ---------------------------------------------------------------------------
# Main repack
# ---------------------------------------------------------------------------

def repack(
    model_type: str,
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
    num_mxfp8_blocks: int | None,
) -> None:
    transformer_dirs = _get_transformer_dirs(model_type)

    print(f"Copying original model to {output_path} (skipping {transformer_dirs}) …")
    shutil.copytree(
        str(original_model_path),
        str(output_path),
        ignore=shutil.ignore_patterns(*transformer_dirs),
    )

    for tdir in transformer_dirs:
        q_subdir = _get_quant_subdir(model_type, quant_path, tdir)
        out_tdir = output_path / tdir
        orig_tdir = original_model_path / tdir
        print(f"\nConverting {tdir} (quant source: {q_subdir.name}) …")
        _convert_transformer(model_type, q_subdir, out_tdir, orig_tdir, num_mxfp8_blocks)

    print(f"\nDone. Merged model → {output_path}")
    print("\nRun inference (quantization auto-detected from config.json):")
    print(f"  python text_to_video.py --model {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--original-model", required=True)
    parser.add_argument("--quant-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--num-mxfp8-blocks",
        type=int,
        default=None,
        help="Number of MXFP8 blocks (0..N-1). Auto-detected from quant_meta if omitted.",
    )
    args = parser.parse_args()

    repack(
        model_type=args.model_type,
        original_model_path=pathlib.Path(args.original_model),
        quant_path=pathlib.Path(args.quant_path),
        output_path=pathlib.Path(args.output_path),
        num_mxfp8_blocks=args.num_mxfp8_blocks,
    )


if __name__ == "__main__":
    main()
