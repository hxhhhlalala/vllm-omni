#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge mixed MXFP8 + W4A4_MXFP4_DUALSCALE quantized Wan2.2 weights into HF Diffusers format.

msModelSlim produces a mixed-precision checkpoint where transformer blocks are split into:
  - Early blocks (0..num_mxfp8_blocks-1): W8A8_MXFP8
  - Remaining blocks (num_mxfp8_blocks..): W4A4_MXFP4_DUALSCALE

MXFP4_DUALSCALE key structure per linear layer
-----------------------------------------------
  blocks.N.X.linear.weight             W4A4_MXFP4_DUALSCALE  – int8 (FP4 packed)
  blocks.N.X.linear.weight_scale       W4A4_MXFP4_DUALSCALE  – uint8 (float8_e8m0fnu fine scale, per-32K)
  blocks.N.X.linear.weight_dual_scale  W4A4_MXFP4_DUALSCALE  – float32 (coarse scale, per-512K)
  blocks.N.X.linear.bias               FLOAT                  – bias (if present)
  blocks.N.X.div.mul_scale             FLOAT                  – float32 per-input-channel activation pre-scale

MXFP8 key structure (no wrapper, same as merge_mxfp8_checkpoint.py):
  blocks.N.X.weight                    W8A8_MXFP8
  blocks.N.X.weight_scale              W8A8_MXFP8

Self-attention QKV notes
------------------------
Self-attention Q/K/V weights are separate in the checkpoint (self_attn.q/k/v) but fused
into a single to_qkv layer in vllm-omni. The transformer's load_weights() handles this via
stacked_params_mapping. This script keeps Q/K/V keys separate — do NOT pre-fuse them here.

For mul_scale specifically: even though Q/K/V process the same input (same mul_scale value),
they are kept separate. load_weights() routes all three to the same to_qkv.mul_scale parameter
and each overwrites the previous. Since Q=K=V for mul_scale, the final value is correct.

NOTE: Pre-fusing as to_qkv.mul_scale would BREAK loading because ".attn1.to_q" is a
substring of ".attn1.to_qkv", causing load_weights() stacked_params_mapping to produce a
garbage key ("to_qkvkv") that is not in params_dict, triggering a break that skips the
direct-load else branch entirely.

Supported model types:
  - Wan2.2-T2V-A14B  (MoE cascade: transformer + transformer_2)
  - Wan2.2-I2V-A14B  (MoE cascade: transformer + transformer_2)
  - Wan2.2-TI2V-5B   (single transformer)

Usage:
  python merge_mxfp4_dualscale_checkpoint.py \\
      --model-type        Wan2.2-T2V-A14B \\
      --original-model    /path/to/Wan2.2-T2V-A14B-Diffusers \\
      --quant-path        /path/to/msmodelslim-output \\
      --output-path       /path/to/merged-output \\
      --num-mxfp8-blocks  5          # auto-detected if omitted
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import warnings
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Key rename: msModelSlim naming → Diffusers / vllm-omni naming
# Identical to merge_mxfp8_checkpoint.py; applied to all blocks uniformly.
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
    # Norm order swap (quant tool: norm1, norm3, norm2 → diffusers: norm1, norm2, norm3)
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

# Suffixes that appear inside .linear.* wrapper for MXFP4 tensors.
_LINEAR_ATTRS = ("weight_dual_scale", "weight_scale", "weight", "bias")

_BLOCK_IDX_RE = re.compile(r"^blocks\.(\d+)\.")


# ---------------------------------------------------------------------------
# Key transformation helpers
# ---------------------------------------------------------------------------


def _parse_block_idx(key: str) -> int | None:
    """Extract block index from a key like 'blocks.5.attn1.to_q.weight'."""
    m = _BLOCK_IDX_RE.match(key)
    return int(m.group(1)) if m else None


def _apply_rename_dict(key: str) -> str:
    for src, dst in TRANSFORMER_KEYS_RENAME_DICT.items():
        key = key.replace(src, dst)
    return key


def _strip_mxfp4_wrapper(key: str) -> str:
    """Strip .linear.ATTR or .div.mul_scale wrappers added by msModelSlim.

    MXFP4 tensors are wrapped in sub-modules:
      X.linear.weight / weight_scale / weight_dual_scale / bias
      X.div.mul_scale

    MXFP8 and FLOAT tensors have no wrappers — this function is a no-op for them.

    Examples after apply_rename_dict:
      attn1.to_q.linear.weight      → attn1.to_q.weight
      attn1.to_q.linear.weight_scale → attn1.to_q.weight_scale
      attn1.to_q.div.mul_scale      → attn1.to_q.mul_scale
      attn1.norm_q.weight           → attn1.norm_q.weight  (unchanged)
    """
    # Check longest attribute names first to avoid partial suffix matches.
    for attr in _LINEAR_ATTRS:
        suffix = f".linear.{attr}"
        if key.endswith(suffix):
            return key[: -len(suffix)] + f".{attr}"
    if key.endswith(".div.mul_scale"):
        return key[: -len(".div.mul_scale")] + ".mul_scale"
    return key


# ---------------------------------------------------------------------------
# Quantization metadata helpers
# ---------------------------------------------------------------------------

# Known quantized weight types (FLOAT tensors don't determine block type).
_MXFP8_TYPE = "mxfp8"
_MXFP4_DUALSCALE_TYPE = "mxfp4_dualscale"


def _classify_blocks(quant_meta: dict[str, str]) -> dict[int, str]:
    """Classify each transformer block by quantization type from quant_meta.

    Returns a dict mapping block_idx → 'mxfp8' | 'mxfp4_dualscale'.
    A block's type is determined by the first quantized (non-FLOAT) tensor found for it.
    """
    block_types: dict[int, str] = {}
    for key, qtype in quant_meta.items():
        idx = _parse_block_idx(key)
        if idx is None or idx in block_types:
            continue
        if qtype.startswith("W8A8_MXFP8"):
            block_types[idx] = _MXFP8_TYPE
        elif qtype.startswith("W4A4_MXFP4_DUALSCALE"):
            block_types[idx] = _MXFP4_DUALSCALE_TYPE
    return block_types


def _print_block_summary(block_types: dict[int, str]) -> None:
    """Print a compact run-length summary of the block layout."""
    if not block_types:
        print("  Block layout: (empty)")
        return

    sorted_indices = sorted(block_types)
    runs: list[tuple[int, int, str]] = []
    run_start = sorted_indices[0]
    run_type = block_types[run_start]
    for idx in sorted_indices[1:]:
        if block_types[idx] != run_type:
            runs.append((run_start, idx - 1, run_type))
            run_start = idx
            run_type = block_types[idx]
    runs.append((run_start, sorted_indices[-1], run_type))

    print(f"  Block layout ({len(sorted_indices)} blocks classified):")
    for start, end, btype in runs:
        count = end - start + 1
        range_str = f"{start}" if start == end else f"{start}–{end}"
        print(f"    blocks {range_str:>8}: {btype}  ({count} block{'s' if count > 1 else ''})")


def _detect_num_mxfp8_blocks(quant_meta: dict[str, str]) -> int:
    """Count leading MXFP8 blocks (blocks 0..N-1).

    Two cases handled:
    - MXFP8 blocks present in quant_meta (W8A8_MXFP8 markers):
      count the consecutive run from block 0.
    - MXFP8 blocks absent from quant_meta (msModelSlim may omit them):
      the index of the first MXFP4_DUALSCALE block equals num_mxfp8_blocks,
      because all blocks before it are implicitly MXFP8.
    """
    block_types = _classify_blocks(quant_meta)
    if not block_types:
        return 0

    sorted_indices = sorted(block_types)
    first_idx = sorted_indices[0]

    if block_types[first_idx] == _MXFP8_TYPE:
        # MXFP8 blocks present: count consecutive run starting at block 0.
        if first_idx != 0:
            warnings.warn(
                f"First classified block is {first_idx} (expected 0); "
                "cannot determine num_mxfp8_blocks reliably. Returning 0."
            )
            return 0
        count = 0
        for idx in sorted_indices:
            if block_types[idx] == _MXFP8_TYPE:
                count += 1
            else:
                break
        return count

    # MXFP8 blocks absent from quant_meta: the first MXFP4 block index is the boundary.
    return first_idx


# ---------------------------------------------------------------------------
# Safetensors I/O
# ---------------------------------------------------------------------------


def _load_safetensors_dir(directory: pathlib.Path, glob: str = "*.safetensors") -> dict[str, torch.Tensor]:
    candidates = sorted(directory.glob(glob))
    if not candidates:
        raise FileNotFoundError(f"No safetensors matching '{glob}' found in {directory}")
    state: dict[str, torch.Tensor] = {}
    for f in candidates:
        state.update(load_file(str(f)))
    return state


def _load_quant_safetensors(directory: pathlib.Path) -> dict[str, torch.Tensor]:
    try:
        return _load_safetensors_dir(directory, "quant_model_weight*.safetensors")
    except FileNotFoundError:
        return _load_safetensors_dir(directory)


def _load_quant_meta(directory: pathlib.Path) -> dict[str, str]:
    candidates = sorted(directory.glob("quant_model_description*.json"))
    if not candidates:
        print(f"  WARNING: No quant_model_description*.json in {directory}; treating all tensors as FLOAT.")
        return {}
    with open(candidates[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-transformer conversion
# ---------------------------------------------------------------------------


def _convert_transformer(
    quant_subdir: pathlib.Path,
    output_dir: pathlib.Path,
    original_transformer_dir: pathlib.Path,
    num_mxfp8_blocks: int | None,
) -> int:
    """Convert one transformer directory. Returns the resolved num_mxfp8_blocks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # BF16 base: ensures non-quantized tensors that msModelSlim might omit are present.
    print(f"  Loading BF16 base from {original_transformer_dir} …")
    base_state = _load_safetensors_dir(original_transformer_dir)
    print(f"  {len(base_state)} BF16 tensors loaded")

    print(f"  Loading quantized weights from {quant_subdir} …")
    quant_state = _load_quant_safetensors(quant_subdir)
    quant_meta = _load_quant_meta(quant_subdir)
    print(f"  {len(quant_state)} quant tensors, {len(quant_meta)} meta entries")

    # Classify blocks and auto-detect / validate num_mxfp8_blocks.
    block_types = _classify_blocks(quant_meta)
    detected = _detect_num_mxfp8_blocks(quant_meta)
    # Fill in inferred MXFP8 blocks (may be absent from quant_meta).
    for i in range(detected):
        block_types.setdefault(i, _MXFP8_TYPE)
    _print_block_summary(block_types)
    if num_mxfp8_blocks is None:
        num_mxfp8_blocks = detected
        print(f"  Auto-detected num_mxfp8_blocks = {num_mxfp8_blocks}")
    elif num_mxfp8_blocks != detected:
        warnings.warn(f"--num-mxfp8-blocks={num_mxfp8_blocks} but auto-detected {detected}. Using the provided value.")

    # Remap all quantized keys.
    # Key transformation is per-block:
    #   MXFP8 blocks        → rename dict only (no .linear./.div. wrappers)
    #   MXFP4_DUALSCALE blocks → rename dict + strip .linear./.div. wrappers
    #   Non-block keys      → rename dict only
    remapped: dict[str, torch.Tensor] = {}
    remapped_meta: dict[str, str] = {}
    skipped: list[str] = []

    for key, tensor in quant_state.items():
        renamed = _apply_rename_dict(key)

        block_idx = _parse_block_idx(renamed)
        if block_idx is not None and block_types.get(block_idx) == _MXFP4_DUALSCALE_TYPE:
            final_key = _strip_mxfp4_wrapper(renamed)
        else:
            final_key = renamed

        # Skip non-tensor metadata keys that msModelSlim sometimes embeds
        # (e.g. quant_type markers stored as scalar tensors).
        if final_key.endswith(".quant_type"):
            skipped.append(key)
            continue

        remapped[final_key] = tensor
        if key in quant_meta:
            remapped_meta[final_key] = quant_meta[key]

    if skipped:
        print(f"  Skipped {len(skipped)} metadata keys (quant_type markers): {skipped[:5]}")

    # Overlay: BF16 base provides the scaffold; quant tensors replace their BF16 counterparts
    # and add the new scale tensors (weight_scale, weight_dual_scale, mul_scale).
    merged = {**base_state, **remapped}

    # Save weights.
    out_weights = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(merged, str(out_weights))
    print(f"  Saved {len(merged)} tensors → {out_weights}")

    # Save remapped quant metadata (for inspection / debugging).
    out_meta_path = output_dir / "quant_model_description.json"
    with open(out_meta_path, "w") as f:
        json.dump(remapped_meta, f, indent=2)

    # Inject quantization_config into config.json.
    src_config = original_transformer_dir / "config.json"
    if src_config.is_file():
        with open(src_config) as f:
            config = json.load(f)
        config["quantization_config"] = _build_quant_config(num_mxfp8_blocks)
        out_config = output_dir / "config.json"
        with open(out_config, "w") as f:
            json.dump(config, f, indent=2)
        print(
            f"  Injected quantization_config "
            f"(mxfp8_mxfp4_dualscale, num_mxfp8_blocks={num_mxfp8_blocks}) "
            f"→ {out_config}"
        )
    else:
        print(f"  WARNING: No config.json at {src_config}; quantization_config not injected.")

    return num_mxfp8_blocks


def _build_quant_config(num_mxfp8_blocks: int) -> dict[str, Any]:
    return {
        "quant_method": "mxfp8_mxfp4_dualscale",
        "num_mxfp8_blocks": num_mxfp8_blocks,
        "is_checkpoint_serialized": True,
    }


# ---------------------------------------------------------------------------
# Model-type helpers
# ---------------------------------------------------------------------------


def _get_transformer_dirs(model_type: str) -> list[str]:
    return ["transformer", "transformer_2"] if model_type in CASCADE_MODEL_TYPES else ["transformer"]


def _get_quant_subdir(model_type: str, quant_path: pathlib.Path, transformer_dir: str) -> pathlib.Path:
    if model_type in CASCADE_MODEL_TYPES:
        sub = "high_noise_model" if transformer_dir == "transformer" else "low_noise_model"
        return quant_path / sub
    return quant_path


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
        resolved_n = _convert_transformer(q_subdir, out_tdir, orig_tdir, num_mxfp8_blocks)
        # Use the resolved value for subsequent transformers in the same cascade.
        num_mxfp8_blocks = resolved_n

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
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES, help="Model variant.")
    parser.add_argument("--original-model", required=True, help="Original HF Diffusers model directory (BF16).")
    parser.add_argument("--quant-path", required=True, help="msModelSlim quantized weights directory.")
    parser.add_argument("--output-path", required=True, help="Output directory for merged model.")
    parser.add_argument(
        "--num-mxfp8-blocks",
        type=int,
        default=None,
        help=("Number of leading MXFP8 blocks (0..N-1). Auto-detected from quant_model_description.json if omitted."),
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
