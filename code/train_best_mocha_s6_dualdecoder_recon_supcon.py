#!/usr/bin/env python3
"""Fine-tune Best Mocha A/B on S6 with reconstruction and motion SupCon."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_best_mocha_s6_full_eval import (
    GRUClassifier,
    GRUTextAnchored,
    Normalizer,
    class_log_prior,
    infer_pair,
    make_loader,
    metrics_dict,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts/care_pd_skeleton6_motion_cache.pt"
DEFAULT_TEMPLATE = ROOT / "artifacts/validation/best_mocha_submission/Best_mocha_submission"
DEFAULT_OUTPUT = ROOT / "artifacts/best_mocha_s6_trueonly_dualdecoder_recon_supcon_tau04"
DEFAULT_ZIP = ROOT / "best_mocha_s6_trueonly_dualdecoder_recon_supcon_logitadj_tau04_submission.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--submission-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--decoder-warmup-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr-core", type=float, default=2e-5)
    parser.add_argument("--lr-aux", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--recon-a-weight", type=float, default=0.05)
    parser.add_argument("--recon-b-weight", type=float, default=0.02)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--label-subset", choices=("true-only", "all"), default="true-only")
    parser.add_argument("--pseudo-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MotionSequenceDecoder(nn.Module):
    """Compact temporal decoder used only while fine-tuning."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128, output_dim: int = 263, max_len: int = 200):
        super().__init__()
        self.max_len = int(max_len)
        self.initial_state = nn.Linear(latent_dim, hidden_dim)
        self.time_tokens = nn.Parameter(torch.empty(1, self.max_len, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        nn.init.normal_(self.time_tokens, mean=0.0, std=0.02)

    def forward(self, embedding: torch.Tensor, frames: int) -> torch.Tensor:
        if frames > self.max_len:
            raise ValueError(f"Decoder received {frames} frames, maximum is {self.max_len}")
        tokens = self.time_tokens[:, :frames].expand(embedding.shape[0], -1, -1)
        hidden = torch.tanh(self.initial_state(embedding)).unsqueeze(0)
        decoded, _ = self.gru(tokens, hidden)
        return self.output(decoded)


class SupConProjection(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(embedding), dim=1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    anchor_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Label-based motion SupCon; no text features are involved."""
    batch_size = features.shape[0]
    if batch_size < 2:
        return features.sum() * 0.0
    logits = features @ features.T / float(temperature)
    identity = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    denominator = ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * denominator
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return features.sum() * 0.0
    per_anchor = -(log_prob * positive).sum(dim=1) / positive_count.clamp_min(1)
    if anchor_weights is None:
        return per_anchor[valid].mean()
    return weighted_mean(per_anchor[valid], anchor_weights[valid])


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    frame_ids = torch.arange(target.shape[1], device=target.device)[None, :]
    valid = frame_ids < lengths.to(target.device)[:, None]
    squared_error = (prediction - target).square() * valid[:, :, None]
    denominator = valid.sum(dim=1).clamp_min(1) * target.shape[-1]
    per_sample = squared_error.sum(dim=(1, 2)) / denominator
    if sample_weights is None:
        return per_sample.mean()
    return weighted_mean(per_sample, sample_weights)


def effective_log_prior(
    labels: torch.Tensor,
    indices: Sequence[int],
    sample_weights: torch.Tensor,
) -> tuple[torch.Tensor, list[float], list[int]]:
    selected = torch.as_tensor(indices, dtype=torch.long)
    selected_labels = labels[selected].long()
    selected_weights = sample_weights[selected].double()
    effective_counts = torch.zeros(4, dtype=torch.float64)
    effective_counts.scatter_add_(0, selected_labels, selected_weights)
    raw_counts = torch.bincount(selected_labels, minlength=4)
    prior = effective_counts / effective_counts.sum().clamp_min(1e-12)
    return (
        prior.clamp_min(1e-12).log(),
        [float(value) for value in effective_counts],
        [int(value) for value in raw_counts],
    )


def set_core_trainable(model_a: nn.Module, model_b: nn.Module, trainable: bool) -> None:
    for parameter in list(model_a.parameters()) + list(model_b.parameters()):
        parameter.requires_grad_(trainable)


def save_submission(template: Path, trained_dir: Path, package_dir: Path, output_zip: Path) -> None:
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_root = package_dir / "Best_mocha_submission"
    shutil.copytree(template, package_root, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc"))
    final_dir = package_root / "weights/final_ensemble"
    for filename in ("model_a.pth", "model_b.pth", "mean.npy", "std.npy", "config.json"):
        shutil.copy2(trained_dir / filename, final_dir / filename)
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_root).as_posix())


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.output_dir / "trainall"
    train_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    motions = torch.as_tensor(payload["motions"])
    lengths = torch.as_tensor(payload["lengths"]).long()
    labels = torch.as_tensor(payload["labels"]).long()
    if len(labels) != 4453:
        raise ValueError(f"Expected all 4,453 Skeleton-6 samples, got {len(labels)}")
    metadata = list(payload["meta"])
    true_mask = torch.as_tensor(
        [str(row.get("label_source", "")).lower() == "true" for row in metadata], dtype=torch.bool,
    )
    if int(true_mask.sum()) != 2953:
        raise ValueError(f"Expected 2,953 true-label Skeleton-6 samples, got {int(true_mask.sum())}")
    sample_weights = torch.where(
        true_mask,
        torch.ones(len(labels), dtype=torch.float64),
        torch.full((len(labels),), float(args.pseudo_weight), dtype=torch.float64),
    )
    if args.label_subset == "true-only":
        indices: Sequence[int] = np.flatnonzero(true_mask.numpy()).astype(np.int64)
    else:
        indices = np.arange(len(labels), dtype=np.int64)

    old_mean = torch.from_numpy(np.load(ROOT / "weights/stats/pdgam/mean.npy")).float()
    old_std = torch.from_numpy(np.load(ROOT / "weights/stats/pdgam/std.npy")).float()
    base_weights = args.template / "weights/final_ensemble"
    pretrained_mean = torch.from_numpy(np.load(base_weights / "mean.npy")).float()
    pretrained_std = torch.from_numpy(np.load(base_weights / "std.npy")).float()
    normalizer = Normalizer(old_mean, old_std, pretrained_mean, pretrained_std)

    model_a = GRUClassifier().to(device)
    model_b = GRUTextAnchored().to(device)
    model_a.load_state_dict(torch.load(base_weights / "model_a.pth", map_location="cpu", weights_only=False), strict=True)
    model_b.load_state_dict(torch.load(base_weights / "model_b.pth", map_location="cpu", weights_only=False), strict=True)
    decoder_a = MotionSequenceDecoder().to(device)
    decoder_b = MotionSequenceDecoder().to(device)
    projection_a = SupConProjection().to(device)
    projection_b = SupConProjection().to(device)

    core_parameters = list(model_a.parameters()) + list(model_b.parameters())
    auxiliary_parameters = (
        list(decoder_a.parameters()) + list(decoder_b.parameters())
        + list(projection_a.parameters()) + list(projection_b.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": core_parameters, "lr": float(args.lr_core)},
            {"params": auxiliary_parameters, "lr": float(args.lr_aux)},
        ],
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    loader = make_loader(motions, lengths, labels, indices, int(args.batch_size), True, int(args.seed))
    history: list[dict[str, float]] = []
    set_core_trainable(model_a, model_b, int(args.decoder_warmup_epochs) == 0)

    for epoch in range(1, int(args.epochs) + 1):
        if epoch == int(args.decoder_warmup_epochs) + 1:
            set_core_trainable(model_a, model_b, True)
            print(f"epoch={epoch}: unfroze both pretrained branches", flush=True)
        model_a.train()
        model_b.train()
        decoder_a.train()
        decoder_b.train()
        projection_a.train()
        projection_b.train()
        totals = np.zeros(8, dtype=np.float64)
        seen = 0
        for batch_motion, batch_lengths, batch_labels, batch_indices in loader:
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_weights = sample_weights[batch_indices].to(device, non_blocking=True)
            motion = normalizer.transform(batch_motion, batch_lengths, device)
            optimizer.zero_grad(set_to_none=True)

            embedding_a = model_a.encode(motion, batch_lengths)
            embedding_b = model_b.encode(motion, batch_lengths)
            logits_a = model_a.classifier(embedding_a)
            logits_b = model_b.classifier(embedding_b)
            ce_a = weighted_mean(F.cross_entropy(logits_a, batch_labels, reduction="none"), batch_weights)
            ce_b = weighted_mean(F.cross_entropy(logits_b, batch_labels, reduction="none"), batch_weights)
            supcon_a = supervised_contrastive_loss(
                projection_a(embedding_a), batch_labels, float(args.supcon_temperature), batch_weights
            )
            supcon_b = supervised_contrastive_loss(
                projection_b(embedding_b), batch_labels, float(args.supcon_temperature), batch_weights
            )
            recon_a = masked_reconstruction_loss(
                decoder_a(embedding_a, motion.shape[1]), motion, batch_lengths, batch_weights
            )
            recon_b = masked_reconstruction_loss(
                decoder_b(embedding_b, motion.shape[1]), motion, batch_lengths, batch_weights
            )
            loss = (
                ce_a + ce_b
                + float(args.supcon_weight) * (supcon_a + supcon_b) / 2.0
                + float(args.recon_a_weight) * recon_a
                + float(args.recon_b_weight) * recon_b
            )
            loss.backward()
            nn.utils.clip_grad_norm_(core_parameters + auxiliary_parameters, 5.0)
            optimizer.step()

            batch_size = int(batch_labels.shape[0])
            seen += batch_size
            totals += np.asarray([
                loss.item(), ce_a.item(), ce_b.item(), supcon_a.item(), supcon_b.item(),
                recon_a.item(), recon_b.item(), optimizer.param_groups[0]["lr"],
            ]) * batch_size
        scheduler.step()
        averages = totals / max(seen, 1)
        row = {
            "epoch": epoch,
            "loss": float(averages[0]),
            "ce_a": float(averages[1]),
            "ce_b": float(averages[2]),
            "supcon_a": float(averages[3]),
            "supcon_b": float(averages[4]),
            "recon_a": float(averages[5]),
            "recon_b": float(averages[6]),
            "lr_core": float(averages[7]),
            "core_frozen": epoch <= int(args.decoder_warmup_epochs),
        }
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == int(args.epochs):
            print(
                f"epoch={epoch:02d}/{args.epochs} loss={row['loss']:.4f} "
                f"ce=({row['ce_a']:.4f},{row['ce_b']:.4f}) "
                f"sup=({row['supcon_a']:.4f},{row['supcon_b']:.4f}) "
                f"recon=({row['recon_a']:.4f},{row['recon_b']:.4f})",
                flush=True,
            )

    model_a.eval()
    model_b.eval()
    log_prior, effective_class_counts, raw_class_counts = effective_log_prior(labels, indices, sample_weights)
    torch.save(model_a.state_dict(), train_dir / "model_a.pth")
    torch.save(model_b.state_dict(), train_dir / "model_b.pth")
    np.save(train_dir / "mean.npy", normalizer.mean.numpy())
    np.save(train_dir / "std.npy", normalizer.std.numpy())
    config = {
        "tau": float(args.tau),
        "log_prior": [float(value) for value in log_prior],
        "class_counts": effective_class_counts,
        "effective_class_counts": effective_class_counts,
        "raw_class_counts": raw_class_counts,
        "target_fps": 25.0,
        "max_len": 200,
        "n_train_walks": len(indices),
        "label_subset": (
            "Skeleton-6 true labels only; all pseudo labels excluded"
            if args.label_subset == "true-only"
            else f"Skeleton-6 all labels; true weight 1.0 and pseudo weight {float(args.pseudo_weight):g}"
        ),
        "pseudo_weight": float(args.pseudo_weight),
        "effective_prior_used": True,
        "initialized_from": "Best_mocha_submission.zip Model A and Model B",
        "training_method": "branch-local decoder reconstruction + label-based motion SupCon + dual CE",
        "text_alignment_used_for_finetuning": False,
        "decoder_training_only": True,
        "loss_weights": {
            "ce_a": 1.0,
            "ce_b": 1.0,
            "supcon_average": float(args.supcon_weight),
            "reconstruction_a": float(args.recon_a_weight),
            "reconstruction_b": float(args.recon_b_weight),
        },
        "supcon_temperature": float(args.supcon_temperature),
        "decoder_warmup_epochs": int(args.decoder_warmup_epochs),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
    }
    (train_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (train_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_a": model_a.state_dict(),
            "model_b": model_b.state_dict(),
            "decoder_a": decoder_a.state_dict(),
            "decoder_b": decoder_b.state_dict(),
            "supcon_projection_a": projection_a.state_dict(),
            "supcon_projection_b": projection_b.state_dict(),
            "config": config,
        },
        train_dir / "training_checkpoint_with_auxiliary_heads.pt",
    )

    eval_args = argparse.Namespace(batch_size=int(args.batch_size), seed=int(args.seed), tau=float(args.tau))
    result = infer_pair(
        model_a=model_a,
        model_b=model_b,
        normalizer=normalizer,
        log_prior=log_prior,
        motions=motions,
        lengths=lengths,
        labels=labels,
        indices=indices,
        args=eval_args,
        device=device,
    )
    train_metrics = {
        "raw": metrics_dict(result["labels"], result["raw_predictions"]),
        "logit_adjusted_tau04": metrics_dict(result["labels"], result["predictions"]),
    }
    (train_dir / "trainall_metrics.json").write_text(json.dumps(train_metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(train_metrics, indent=2), flush=True)

    save_submission(
        args.template,
        train_dir,
        args.output_dir / "submission_package",
        args.submission_zip,
    )
    print(f"submission={args.submission_zip}", flush=True)


if __name__ == "__main__":
    train(parse_args())
