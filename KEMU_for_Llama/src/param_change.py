#!/usr/bin/env python3

import argparse
import gc
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "safetensors is required for parameter change statistics."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute parameter change statistics between a reference and target HF checkpoint."
    )
    parser.add_argument("task_name", type=str, help="Experiment task name.")
    parser.add_argument(
        "--reference_model_dir",
        type=Path,
        required=True,
        help="Path to the reference checkpoint directory.",
    )
    parser.add_argument(
        "--target_model_dir",
        type=Path,
        required=True,
        help="Path to the target checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the parameter statistics.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Threshold used for changed-ratio statistics.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many embedding rows with the largest deltas to keep.",
    )
    return parser.parse_args()


def _load_index_map(model_dir: Path):
    st_index = model_dir / "model.safetensors.index.json"
    if st_index.exists():
        with st_index.open("r", encoding="utf-8") as fh:
            return "safetensors", json.load(fh)["weight_map"]

    st_single = model_dir / "model.safetensors"
    if st_single.exists():
        with safe_open(str(st_single), framework="pt", device="cpu") as handle:
            return "safetensors", {key: st_single.name for key in handle.keys()}

    pt_index = model_dir / "pytorch_model.bin.index.json"
    if pt_index.exists():
        with pt_index.open("r", encoding="utf-8") as fh:
            return "pt", json.load(fh)["weight_map"]

    pt_single = model_dir / "pytorch_model.bin"
    if pt_single.exists():
        state = torch.load(pt_single, map_location="cpu")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        return "pt", {key: pt_single.name for key in state.keys()}

    raise FileNotFoundError(f"No supported checkpoint files found under {model_dir}")


class CheckpointReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.format, self.weight_map = _load_index_map(model_dir)
        self.keys = set(self.weight_map.keys())
        self._stack: Optional[ExitStack] = None
        self._handles = {}
        self._cached_file = None
        self._cached_state = None

        if self.format == "safetensors":
            self._stack = ExitStack()
            for rel_name in sorted(set(self.weight_map.values())):
                path = self.model_dir / rel_name
                self._handles[rel_name] = self._stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )

    def get_tensor(self, key: str) -> torch.Tensor:
        rel_name = self.weight_map[key]
        if self.format == "safetensors":
            return self._handles[rel_name].get_tensor(key)

        if rel_name != self._cached_file:
            state = torch.load(self.model_dir / rel_name, map_location="cpu")
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            self._cached_file = rel_name
            self._cached_state = state
        return self._cached_state[key]

    def close(self):
        if self._stack is not None:
            self._stack.close()
        self._cached_state = None
        gc.collect()


def module_group(param_name: str) -> str:
    name = param_name.lower()
    if any(token in name for token in ("embed_tokens", "tok_embeddings", "embedding", "wte")):
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if any(token in name for token in ("self_attn", ".attn", "attention")):
        return "attention"
    if any(token in name for token in ("mlp", "feed_forward", "ffn")):
        return "mlp"
    return "other"


def new_bucket():
    return {
        "ref_sq": 0.0,
        "diff_sq": 0.0,
        "abs_sum": 0.0,
        "numel": 0,
        "changed": 0,
        "max_abs": 0.0,
    }


def update_bucket(bucket: Dict[str, float], ref_tensor: torch.Tensor, diff_tensor: torch.Tensor, epsilon: float):
    ref_f = ref_tensor.detach().to(torch.float32)
    diff_f = diff_tensor.detach().to(torch.float32)
    abs_diff = diff_f.abs()

    bucket["ref_sq"] += float((ref_f * ref_f).sum().item())
    bucket["diff_sq"] += float((diff_f * diff_f).sum().item())
    bucket["abs_sum"] += float(abs_diff.sum().item())
    bucket["numel"] += diff_f.numel()
    bucket["changed"] += int((abs_diff > epsilon).sum().item())
    bucket["max_abs"] = max(bucket["max_abs"], float(abs_diff.max().item()))


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    ref_l2 = math.sqrt(bucket["ref_sq"])
    diff_l2 = math.sqrt(bucket["diff_sq"])
    numel = max(int(bucket["numel"]), 1)
    return {
        "abs_l2": diff_l2,
        "rel_l2": diff_l2 / ref_l2 if ref_l2 > 0 else 0.0,
        "abs_l1": bucket["abs_sum"],
        "mean_abs_delta": bucket["abs_sum"] / numel,
        "changed_ratio": bucket["changed"] / numel,
        "max_abs_delta": bucket["max_abs"],
        "numel": int(bucket["numel"]),
    }


def is_input_embedding(name: str, tensor: torch.Tensor) -> bool:
    lowered = name.lower()
    if tensor.ndim != 2:
        return False
    return any(
        token in lowered
        for token in ("model.embed_tokens.weight", "embed_tokens.weight", "tok_embeddings.weight", "wte.weight")
    )


def collect_top_rows(row_norms: torch.Tensor, topk: int):
    if row_norms.numel() == 0:
        return []
    limit = min(topk, row_norms.numel())
    values, indices = torch.topk(row_norms, k=limit)
    return [
        {"row": int(idx.item()), "delta_l2": float(val.item())}
        for idx, val in zip(indices, values)
    ]


def compute_stats(
    reference_dir: Path,
    target_dir: Path,
    epsilon: float,
    topk: int,
):
    ref_reader = CheckpointReader(reference_dir)
    tgt_reader = CheckpointReader(target_dir)

    try:
        common_keys = sorted(ref_reader.keys & tgt_reader.keys)
        missing_in_reference = sorted(tgt_reader.keys - ref_reader.keys)
        missing_in_target = sorted(ref_reader.keys - tgt_reader.keys)

        total_bucket = new_bucket()
        module_buckets = {name: new_bucket() for name in ("embedding", "attention", "mlp", "lm_head", "other")}

        embedding_row_count = 0
        embedding_row_changed = 0
        embedding_row_max = 0.0
        embedding_row_top = []

        for key in common_keys:
            ref_tensor = ref_reader.get_tensor(key)
            tgt_tensor = tgt_reader.get_tensor(key)

            if ref_tensor.shape != tgt_tensor.shape:
                raise ValueError(f"Shape mismatch for {key}: {ref_tensor.shape} vs {tgt_tensor.shape}")

            diff = tgt_tensor.detach().to(torch.float32) - ref_tensor.detach().to(torch.float32)
            update_bucket(total_bucket, ref_tensor, diff, epsilon)
            update_bucket(module_buckets[module_group(key)], ref_tensor, diff, epsilon)

            if is_input_embedding(key, diff):
                row_norms = diff.norm(dim=1)
                embedding_row_count += row_norms.numel()
                embedding_row_changed += int((row_norms > epsilon).sum().item())
                embedding_row_max = max(embedding_row_max, float(row_norms.max().item()))
                if topk > 0:
                    embedding_row_top = collect_top_rows(row_norms, topk)

            del ref_tensor
            del tgt_tensor
            del diff

        stats = {
            "common_tensor_count": len(common_keys),
            "missing_in_reference_count": len(missing_in_reference),
            "missing_in_target_count": len(missing_in_target),
            "epsilon": epsilon,
            "all_abs_l2": 0.0,
            "all_rel_l2": 0.0,
            "all_abs_l1": 0.0,
            "all_mean_abs_delta": 0.0,
            "all_changed_ratio": 0.0,
            "all_max_abs_delta": 0.0,
            "all_numel": 0,
            "embedding_row_count": embedding_row_count,
            "embedding_row_changed": embedding_row_changed,
            "embedding_row_changed_ratio": (
                embedding_row_changed / embedding_row_count if embedding_row_count > 0 else 0.0
            ),
            "embedding_row_max_delta": embedding_row_max,
            "embedding_row_top": embedding_row_top,
            "missing_in_reference_sample": missing_in_reference[:20],
            "missing_in_target_sample": missing_in_target[:20],
        }

        total_final = finalize_bucket(total_bucket)
        stats.update({f"all_{key}": value for key, value in total_final.items()})

        for group_name, bucket in module_buckets.items():
            final_bucket = finalize_bucket(bucket)
            for key, value in final_bucket.items():
                stats[f"module_{group_name}_{key}"] = value

        return stats
    finally:
        ref_reader.close()
        tgt_reader.close()


def main():
    args = parse_args()
    stats = compute_stats(
        reference_dir=args.reference_model_dir,
        target_dir=args.target_model_dir,
        epsilon=args.epsilon,
        topk=args.topk,
    )

    payload = {
        "task_name": args.task_name,
        "reference_model_dir": str(args.reference_model_dir),
        "target_model_dir": str(args.target_model_dir),
        "parameter_change": stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Parameter change stats written to {args.output}")


if __name__ == "__main__":
    main()
