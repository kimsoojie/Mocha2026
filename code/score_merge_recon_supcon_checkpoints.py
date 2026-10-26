#!/usr/bin/env python3
"""SCORE-merge CARE-PD recon_supcon source checkpoints.

For 2-D tensors this implements the SCORE procedure from
"Bridging Domains through Subspace-Aware Model Merging":
SVD each source delta, build a shared orthogonal singular basis, project deltas
into that basis, prune off-diagonal outliers, and reconstruct one merged delta.
For non-2-D tensors, the script falls back to summing deltas from the shared
base checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-motion-encoder", type=Path, default=ROOT / "weights/backbone/motion_encoder_finetuned.pth")
    parser.add_argument(
        "--base-classifier",
        type=Path,
        default=ROOT / "weights/classifier/care_pd_skeleton6_smpl_baseline_trainall_classifier.pth",
    )
    parser.add_argument("--source-motion-encoders", type=Path, nargs="+", required=True)
    parser.add_argument("--source-classifiers", type=Path, nargs="+", required=True)
    parser.add_argument("--source-names", nargs="+", required=True)
    parser.add_argument("--output-motion-encoder", type=Path, required=True)
    parser.add_argument("--output-classifier", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=1.96)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale applied to the final merged delta before adding it to the base checkpoint.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected state dict, got {type(obj)!r}")
    return obj


def components_per_model(shape: torch.Size, num_models: int) -> int:
    min_dim = int(min(shape))
    return max(1, min_dim // max(1, num_models))


def orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right_h


def trim_offdiag(projected: torch.Tensor, tau: float) -> tuple[torch.Tensor, float]:
    r = projected.shape[0]
    eye = torch.eye(r, dtype=torch.bool, device=projected.device)
    off_vals = projected[~eye]
    keep = eye.clone()
    pruned_fraction = 0.0
    if off_vals.numel() > 0:
        mu = off_vals.mean()
        sigma = off_vals.std(unbiased=False)
        if float(sigma.item()) > 1e-12:
            keep_off = (projected - mu).abs() < float(tau) * sigma
            keep = eye | (keep_off & ~eye)
            pruned_fraction = float((~keep & ~eye).sum().item() / max(1, (~eye).sum().item()))
        else:
            pruned_fraction = 1.0
    return torch.where(keep, projected, torch.zeros_like(projected)), pruned_fraction


def score_merge_tensor(
    base: torch.Tensor,
    sources: list[torch.Tensor],
    *,
    tau: float,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not torch.is_floating_point(base):
        return sources[0].detach().cpu(), {"method": "copy_first_non_float"}

    if base.ndim != 2:
        delta = sum((src.to(device=device, dtype=torch.float32) - base.to(device=device, dtype=torch.float32)) for src in sources)
        merged = base.to(device=device, dtype=torch.float32) + float(scale) * delta
        return merged.to(dtype=base.dtype).cpu(), {"method": "sum_delta_non_2d"}

    num_models = len(sources)
    k = components_per_model(base.shape, num_models)
    base_f = base.to(device=device, dtype=torch.float32)
    deltas = [src.to(device=device, dtype=torch.float32) - base_f for src in sources]

    u_parts = []
    v_parts = []
    for delta in deltas:
        u, _, vh = torch.linalg.svd(delta, full_matrices=False)
        u_parts.append(u[:, :k])
        v_parts.append(vh.transpose(0, 1)[:, :k])

    u_star = torch.cat(u_parts, dim=1)
    v_star = torch.cat(v_parts, dim=1)
    u_perp = orthogonalize(u_star)
    v_perp = orthogonalize(v_star)
    r = min(u_perp.shape[1], v_perp.shape[1])
    u_perp = u_perp[:, :r]
    v_perp = v_perp[:, :r]

    sigma_score = torch.zeros((r, r), dtype=torch.float32, device=device)
    pruned = []
    for delta in deltas:
        projected = u_perp.transpose(0, 1) @ delta @ v_perp
        trimmed, pruned_fraction = trim_offdiag(projected, tau)
        sigma_score = sigma_score + trimmed
        pruned.append(pruned_fraction)

    merged_delta = u_perp @ sigma_score @ v_perp.transpose(0, 1)
    merged = base_f + float(scale) * merged_delta
    info = {
        "method": "score_svd_shared_basis",
        "components_per_model": int(k),
        "shared_rank": int(r),
        "offdiag_pruned_fraction_mean": float(sum(pruned) / max(1, len(pruned))),
        "offdiag_pruned_fraction_by_source": pruned,
    }
    return merged.to(dtype=base.dtype).cpu(), info


def score_merge_state(
    base_state: dict[str, torch.Tensor],
    source_states: list[dict[str, torch.Tensor]],
    *,
    tau: float,
    scale: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    merged: dict[str, torch.Tensor] = {}
    key_info: dict[str, Any] = {}
    for key, base_value in base_state.items():
        if key not in source_states[0]:
            raise KeyError(f"{key} missing from source state")
        source_values = [state[key] for state in source_states]
        merged_value, info = score_merge_tensor(base_value, source_values, tau=tau, scale=scale, device=device)
        merged[key] = merged_value
        key_info[key] = info
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return merged, key_info


def validate_args(args: argparse.Namespace) -> None:
    if len(args.source_motion_encoders) != len(args.source_classifiers):
        raise ValueError("--source-motion-encoders and --source-classifiers must have the same length")
    if len(args.source_motion_encoders) != len(args.source_names):
        raise ValueError("--source-names must match the number of source checkpoints")
    if len(args.source_motion_encoders) < 2:
        raise ValueError("SCORE merge requires at least two source models")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"SCORE merging {len(args.source_names)} source model(s) on {device}")
    print(f"Sources: {args.source_names}")

    base_motion = unwrap_state(torch_load(args.base_motion_encoder, "cpu"))
    source_motions = [unwrap_state(torch_load(path, "cpu")) for path in args.source_motion_encoders]
    merged_motion, motion_info = score_merge_state(
        base_motion,
        source_motions,
        tau=float(args.tau),
        scale=float(args.scale),
        device=device,
    )

    base_classifier = unwrap_state(torch_load(args.base_classifier, "cpu"))
    source_classifiers = [unwrap_state(torch_load(path, "cpu")) for path in args.source_classifiers]
    merged_classifier, classifier_info = score_merge_state(
        base_classifier,
        source_classifiers,
        tau=float(args.tau),
        scale=float(args.scale),
        device=device,
    )

    args.output_motion_encoder.parent.mkdir(parents=True, exist_ok=True)
    args.output_classifier.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_motion, args.output_motion_encoder)
    torch.save(merged_classifier, args.output_classifier)

    metrics = {
        "method": "SCORE",
        "paper": "Bridging Domains through Subspace-Aware Model Merging, arXiv:2603.05768v2",
        "source_names": list(args.source_names),
        "base_motion_encoder": str(args.base_motion_encoder),
        "base_classifier": str(args.base_classifier),
        "source_motion_encoders": [str(p) for p in args.source_motion_encoders],
        "source_classifiers": [str(p) for p in args.source_classifiers],
        "output_motion_encoder": str(args.output_motion_encoder),
        "output_classifier": str(args.output_classifier),
        "tau": float(args.tau),
        "scale": float(args.scale),
        "device": str(device),
        "motion_encoder_keys": motion_info,
        "classifier_keys": classifier_info,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved merged motion encoder to {args.output_motion_encoder}")
    print(f"Saved merged classifier to {args.output_classifier}")
    print(f"Saved merge metrics to {args.metrics}")


if __name__ == "__main__":
    main()
