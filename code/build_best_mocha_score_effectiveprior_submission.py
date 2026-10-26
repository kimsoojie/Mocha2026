#!/usr/bin/env python3
"""Build a Best-MoCHA SCORE-merged submission from the original package."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-zip", type=Path, default=ROOT / "Best_mocha_submission.zip")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--effective-counts", type=float, nargs=4, required=True)
    parser.add_argument("--logit-tau", type=float, default=0.4)
    parser.add_argument(
        "--prior-source",
        default="Skeleton-6 labels with true weight 1.0 and pseudo weight 0.3",
    )
    parser.add_argument("--pseudo-weight", type=float, default=0.3)
    parser.add_argument("--prior-sample-count", type=int, default=None)
    parser.add_argument(
        "--prior-kind",
        choices=("effective", "true_label", "none"),
        default="effective",
    )
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--selection-note", required=True)
    parser.add_argument("--score-tau", type=float, default=1.96)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--source-finetune-epochs", type=int, default=30)
    parser.add_argument("--source-lr-backbone", type=float, default=None)
    parser.add_argument("--source-lr-head", type=float, default=None)
    parser.add_argument("--source-lr-aux", type=float, default=None)
    parser.add_argument(
        "--ce-class-weights",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("W0", "W1", "W2", "W3"),
        help="CE class weights used for both source-fine-tuned branches.",
    )
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument(
        "--source-sample-counts",
        type=int,
        nargs=4,
        metavar=("PDGAM", "BMCLAB", "TSDUPD", "GAIT3D"),
        default=(2643, 1166, 511, 133),
    )
    parser.add_argument(
        "--training-method",
        default="four S7 source-specific full fine-tunes with dual CE, branch-local reconstruction, and motion SupCon; SCORE-merged A/B deltas",
    )
    parser.add_argument("--caption-source", default=None)
    parser.add_argument("--caption-encoder", default=None)
    parser.add_argument("--caption-matched", type=int, default=None)
    parser.add_argument("--caption-missing", type=int, default=None)
    parser.add_argument(
        "--label-subset-description",
        default=None,
        help="Optional exact training-label description stored in config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    with zipfile.ZipFile(args.template_zip) as archive:
        archive.extractall(args.work_dir)
    nested_package_root = args.work_dir / "Best_mocha_submission"
    package_root = nested_package_root if nested_package_root.is_dir() else args.work_dir
    final_dir = package_root / "weights/final_ensemble"
    shutil.copy2(args.model_a, final_dir / "model_a.pth")
    shutil.copy2(args.model_b, final_dir / "model_b.pth")

    counts = [float(value) for value in args.effective_counts]
    total = sum(counts)
    no_adjustment = args.prior_kind == "none"
    log_prior = (
        [0.0, 0.0, 0.0, 0.0]
        if no_adjustment
        else [__import__("math").log(max(value / total, 1e-12)) for value in counts]
    )
    config = json.loads((final_dir / "config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "tau": 0.0 if no_adjustment else float(args.logit_tau),
            "log_prior": log_prior,
            "class_counts": counts,
            "effective_class_counts": counts,
            "raw_class_counts": [int(round(value)) for value in counts],
            "effective_prior_used": args.prior_kind == "effective",
            "effective_prior_source": "none" if no_adjustment else args.prior_source,
            "posthoc_prior_kind": args.prior_kind,
            "posthoc_prior_sample_count": (
                0 if no_adjustment else int(args.prior_sample_count or round(total))
            ),
            "pseudo_weight": float(args.pseudo_weight),
            "n_train_walks": int(sum(args.source_sample_counts)),
            "label_subset": args.label_subset_description
            or (
                "Skeleton-7 source buckets, true labels only; pseudo labels excluded"
                if args.prior_kind == "true_label"
                else "Skeleton-7 assigned buckets, true and pseudo labels included"
            ),
            "score_merged": True,
            "score_tau": float(args.score_tau),
            "score_scale": float(args.score_scale),
            "initialized_from": f"{args.template_zip.name} Model A and Model B",
            "training_method": args.training_method,
            "lambda_contrast": float(args.supcon_weight),
            "text_alignment_used_for_finetuning": args.caption_encoder is not None,
            "decoder_training_only": True,
            "loss_weights": {
                "ce_a": 1.0,
                "ce_b": 1.0,
                "ce_class_weights": [float(value) for value in args.ce_class_weights],
                "supcon_a": float(args.supcon_weight),
                "supcon_b": float(args.supcon_weight),
                "reconstruction_a": 0.05,
                "reconstruction_b": 0.02,
            },
            "source_finetune_epochs": int(args.source_finetune_epochs),
            "supcon_temperature": float(args.supcon_temperature),
            "source_learning_rates": {
                "backbone": args.source_lr_backbone,
                "heads": args.source_lr_head,
                "decoder_and_clip_adapter": args.source_lr_aux,
            },
            "source_sample_counts": dict(
                zip(
                    ("PD-GaM", "BMCLab", "T-SDU-PD", "3DGait"),
                    (int(value) for value in args.source_sample_counts),
                )
            ),
            "selection_note": args.selection_note,
        }
    )
    if args.caption_encoder is not None:
        config["loss_weights"]["clip_caption_contrast_b"] = 1.0
        config["clip_caption_temperature"] = 0.07
        config["clip_frozen"] = True
        config["caption_source"] = args.caption_source
        config["caption_encoder"] = args.caption_encoder
        config["caption_matched"] = args.caption_matched
        config["caption_missing"] = args.caption_missing
    (final_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for path in args.work_dir.rglob("__MACOSX"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in args.work_dir.rglob(".DS_Store"):
        path.unlink(missing_ok=True)
    for path in args.work_dir.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in args.work_dir.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    for path in package_root.rglob("predictions.json"):
        path.unlink(missing_ok=True)

    args.output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_root).as_posix())
    with zipfile.ZipFile(args.output_zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP entry: {bad}")
    print(f"saved={args.output_zip}")
    print(f"effective_counts={counts}")


if __name__ == "__main__":
    main()
