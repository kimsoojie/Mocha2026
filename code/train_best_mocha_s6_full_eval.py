#!/usr/bin/env python3
"""Retrain the Best Mocha two-BiGRU method on Skeleton-6 and evaluate it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts/care_pd_skeleton6_motion_cache.pt"
DEFAULT_BGE = ROOT / "artifacts/models/bge-small-en-v1.5"
DEFAULT_TEMPLATE = ROOT / "artifacts/validation/best_mocha_submission/Best_mocha_submission"
DEFAULT_OUTPUT = ROOT / "artifacts/best_mocha_s6_retrained"
DEFAULT_ZIP = ROOT / "best_mocha_s6_trainall_submission.zip"

TRUE_SOURCES = ("PD-GaM", "BMCLab", "T-SDU-PD", "3DGait")
LABEL_COLORS = {0: "#2563EB", 1: "#16A34A", 2: "#F59E0B", 3: "#DC2626"}
CLASS_TEXTS = (
    "Normal Parkinson gait score zero: natural walking speed and stride length, clear foot clearance "
    "and heel strike, regular steps, normal arm swing, upright stable trunk and smooth turning.",
    "Mild Parkinson gait score one: independent stable walking with slightly reduced speed and stride, "
    "mildly reduced foot clearance or heel strike, slight arm swing asymmetry and mildly slower turning.",
    "Moderate Parkinson gait score two: clearly slow independent gait with persistently short steps, "
    "reduced foot clearance, shuffling, reduced knee flexion and arm swing, trunk flexion and multi-step turning.",
    "Severe Parkinson gait score three: extremely short and irregular steps, severe shuffling or foot dragging, "
    "almost absent arm swing, marked trunk instability, freezing and very difficult unsafe turning.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bge-model", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--submission-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-contrast", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_name(meta: dict[str, Any]) -> str:
    value = str(meta.get("source", "unknown"))
    return value.removesuffix("_canonical.pkl").removesuffix(".pkl")


class GRUClassifier(nn.Module):
    """Exact Model A architecture from Best_mocha_submission.zip."""

    def __init__(self, input_dim: int = 263, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 4),
        )

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.gru(packed)
        return torch.cat([hidden[0], hidden[1]], dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, lengths))


class GRUTextAnchored(nn.Module):
    """Exact Model B architecture, including its training-only text projection."""

    def __init__(self, input_dim: int = 263, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 4),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Linear(128, 384),
        )

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.gru(packed)
        return torch.cat([hidden[0], hidden[1]], dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, lengths))


class IndexedCacheDataset(Dataset):
    def __init__(self, motions: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor, indices: Sequence[int]):
        self.motions = motions
        self.lengths = lengths
        self.labels = labels
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = self.indices[position]
        return self.motions[index], self.lengths[index], self.labels[index], index


def sorted_collate(
    rows: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    motions = torch.stack([row[0] for row in rows])
    lengths = torch.stack([row[1] for row in rows]).long()
    labels = torch.stack([row[2] for row in rows]).long()
    indices = torch.as_tensor([row[3] for row in rows], dtype=torch.long)
    order = torch.argsort(lengths, descending=True)
    return motions[order], lengths[order], labels[order], indices[order]


def make_loader(
    motions: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        IndexedCacheDataset(motions, lengths, labels, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=sorted_collate,
    )


@dataclass
class Normalizer:
    old_mean: torch.Tensor
    old_std: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, motions: torch.Tensor, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        motions = motions.to(device=device, dtype=torch.float32, non_blocking=True)
        lengths_device = lengths.to(device=device, non_blocking=True)
        old_mean = self.old_mean.to(device)
        old_std = self.old_std.to(device)
        mean = self.mean.to(device)
        std = self.std.to(device)
        raw = motions * old_std[None, None, :] + old_mean[None, None, :]
        normalized = (raw - mean[None, None, :]) / std[None, None, :]
        valid = torch.arange(normalized.shape[1], device=device)[None, :] < lengths_device[:, None]
        return normalized.masked_fill(~valid[:, :, None], 0.0)


def fit_normalizer(
    motions: torch.Tensor,
    lengths: torch.Tensor,
    train_indices: Sequence[int],
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    chunk_size: int = 128,
) -> Normalizer:
    total = torch.zeros(263, dtype=torch.float64)
    total_sq = torch.zeros(263, dtype=torch.float64)
    count = 0
    for start in range(0, len(train_indices), chunk_size):
        index = torch.as_tensor(train_indices[start : start + chunk_size], dtype=torch.long)
        normalized = motions[index].float()
        raw = normalized * old_std[None, None, :] + old_mean[None, None, :]
        lens = lengths[index].long()
        valid = torch.arange(raw.shape[1])[None, :] < lens[:, None]
        values = raw[valid]
        total += values.double().sum(dim=0)
        total_sq += values.double().square().sum(dim=0)
        count += int(values.shape[0])
    mean = total / max(count, 1)
    variance = total_sq / max(count, 1) - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return Normalizer(old_mean.float(), old_std.float(), mean.float(), std.float())


def create_text_anchors(model_path: Path, output_dir: Path) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(model_path), device="cpu")
    embeddings = encoder.encode(
        list(CLASS_TEXTS), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    if embeddings.shape != (4, 384):
        raise ValueError(f"Expected four 384-D BGE anchors, got {embeddings.shape}")
    np.save(output_dir / "bge_class_anchors.npy", embeddings)
    (output_dir / "class_anchor_texts.json").write_text(
        json.dumps(dict(enumerate(CLASS_TEXTS)), indent=2) + "\n", encoding="utf-8",
    )
    del encoder
    return torch.from_numpy(embeddings)


def class_log_prior(labels: torch.Tensor, indices: Sequence[int]) -> tuple[torch.Tensor, list[int]]:
    counts = torch.bincount(labels[torch.as_tensor(indices)].long(), minlength=4)
    prior = counts.float() / counts.sum().clamp_min(1)
    return prior.clamp_min(1e-12).log(), [int(value) for value in counts]


def harmonic_mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers or any(value <= 0 for value in numbers):
        return 0.0
    return len(numbers) / sum(1.0 / value for value in numbers)


def metrics_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def train_pair(
    *,
    run_name: str,
    motions: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    train_indices: Sequence[int],
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    text_anchors: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    seed_offset: int,
) -> tuple[GRUClassifier, GRUTextAnchored, Normalizer, torch.Tensor, list[int]]:
    seed = int(args.seed) + int(seed_offset)
    set_seed(seed)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    normalizer = fit_normalizer(motions, lengths, train_indices, old_mean, old_std)
    log_prior, counts = class_log_prior(labels, train_indices)

    model_a = GRUClassifier().to(device)
    model_b = GRUTextAnchored().to(device)
    optimizer = torch.optim.AdamW(
        list(model_a.parameters()) + list(model_b.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    anchors = F.normalize(text_anchors.to(device), dim=1)
    loader = make_loader(
        motions, lengths, labels, train_indices, int(args.batch_size), True, seed,
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model_a.train()
        model_b.train()
        totals = np.zeros(5, dtype=np.float64)
        seen = 0
        for batch_motion, batch_lengths, batch_labels, _ in loader:
            batch_labels = batch_labels.to(device, non_blocking=True)
            motion = normalizer.transform(batch_motion, batch_lengths, device)
            optimizer.zero_grad(set_to_none=True)

            embedding_a = model_a.encode(motion, batch_lengths)
            embedding_b = model_b.encode(motion, batch_lengths)
            logits_a = model_a.classifier(embedding_a)
            logits_b = model_b.classifier(embedding_b)
            anchor_logits = F.normalize(model_b.text_proj(embedding_b), dim=1) @ anchors.T
            anchor_logits = anchor_logits / float(args.temperature)

            loss_a = F.cross_entropy(logits_a, batch_labels)
            loss_b = F.cross_entropy(logits_b, batch_labels)
            loss_anchor = F.cross_entropy(anchor_logits, batch_labels)
            loss = loss_a + loss_b + float(args.lambda_contrast) * loss_anchor
            loss.backward()
            nn.utils.clip_grad_norm_(list(model_a.parameters()) + list(model_b.parameters()), 5.0)
            optimizer.step()

            batch_n = int(batch_labels.shape[0])
            seen += batch_n
            totals += np.asarray(
                [loss.item(), loss_a.item(), loss_b.item(), loss_anchor.item(), optimizer.param_groups[0]["lr"]]
            ) * batch_n
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "loss": float(totals[0] / seen),
            "ce_a": float(totals[1] / seen),
            "ce_b": float(totals[2] / seen),
            "text_anchor": float(totals[3] / seen),
            "lr": float(totals[4] / seen),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(args.epochs):
            print(
                f"[{run_name}] epoch {epoch:02d}/{args.epochs}: loss={row['loss']:.4f} "
                f"ceA={row['ce_a']:.4f} ceB={row['ce_b']:.4f} anchor={row['text_anchor']:.4f}",
                flush=True,
            )

    torch.save(model_a.state_dict(), run_dir / "model_a.pth")
    torch.save(model_b.state_dict(), run_dir / "model_b.pth")
    np.save(run_dir / "mean.npy", normalizer.mean.numpy())
    np.save(run_dir / "std.npy", normalizer.std.numpy())
    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    config = {
        "tau": float(args.tau),
        "log_prior": [float(value) for value in log_prior],
        "class_counts": counts,
        "target_fps": 25.0,
        "max_len": 200,
        "lambda_contrast": float(args.lambda_contrast),
        "contrast_temperature": float(args.temperature),
        "n_train_walks": len(train_indices),
        "seed": seed,
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "training_method": "CE Model A + CE Model B + BGE-small class-anchor contrastive loss",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    model_a.eval()
    model_b.eval()
    return model_a, model_b, normalizer, log_prior, counts


def infer_pair(
    *,
    model_a: GRUClassifier,
    model_b: GRUTextAnchored,
    normalizer: Normalizer,
    log_prior: torch.Tensor,
    motions: torch.Tensor,
    lengths: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    loader = make_loader(motions, lengths, labels, indices, int(args.batch_size), False, int(args.seed))
    records: dict[int, tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for batch_motion, batch_lengths, batch_labels, batch_indices in loader:
            motion = normalizer.transform(batch_motion, batch_lengths, device)
            batch_labels_device = batch_labels.to(device)
            embedding_a = model_a.encode(motion, batch_lengths)
            embedding_b = model_b.encode(motion, batch_lengths)
            logits_a = model_a.classifier(embedding_a)
            logits_b = model_b.classifier(embedding_b)
            probs = (F.softmax(logits_a, dim=1) + F.softmax(logits_b, dim=1)) / 2.0
            raw_pred = probs.argmax(dim=1)
            adjusted_scores = torch.log(probs.clamp_min(1e-12)) - float(args.tau) * log_prior.to(device)[None, :]
            adjusted_pred = adjusted_scores.argmax(dim=1)
            fused = torch.cat([F.normalize(embedding_a, dim=1), F.normalize(embedding_b, dim=1)], dim=1)
            for row in range(len(batch_indices)):
                index = int(batch_indices[row])
                records[index] = (
                    int(batch_labels_device[row]),
                    int(raw_pred[row]),
                    int(adjusted_pred[row]),
                    embedding_a[row].cpu().numpy().astype(np.float32),
                    embedding_b[row].cpu().numpy().astype(np.float32),
                    fused[row].cpu().numpy().astype(np.float32),
                )

    ordered = [records[int(index)] for index in indices]
    return {
        "indices": np.asarray(indices, dtype=np.int64),
        "labels": np.asarray([row[0] for row in ordered], dtype=np.int64),
        "raw_predictions": np.asarray([row[1] for row in ordered], dtype=np.int64),
        "predictions": np.asarray([row[2] for row in ordered], dtype=np.int64),
        "model_a_embeddings": np.stack([row[3] for row in ordered]),
        "model_b_embeddings": np.stack([row[4] for row in ordered]),
        "fused_embeddings": np.stack([row[5] for row in ordered]),
    }


def run_tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    value = min(float(perplexity), max(5.0, (len(features) - 1) / 3.0))
    kwargs = dict(
        n_components=2,
        perplexity=value,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        return TSNE(max_iter=1200, **kwargs).fit_transform(features).astype(np.float32)
    except TypeError:
        return TSNE(n_iter=1200, **kwargs).fit_transform(features).astype(np.float32)


def scatter_labels(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray) -> None:
    for label in range(4):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.65, c=LABEL_COLORS[label], label=f"Label {label}")
    ax.legend(fontsize=8, markerscale=1.4)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def save_trainall_tsne(result: dict[str, Any], meta: Sequence[dict[str, Any]], output: Path, args: argparse.Namespace) -> None:
    coords = run_tsne(result["fused_embeddings"], int(args.seed), float(args.tsne_perplexity))
    labels = result["labels"]
    indices = result["indices"]
    sources = np.asarray([source_name(meta[int(index)]) for index in indices], dtype=object)
    label_sources = np.asarray([str(meta[int(index)].get("label_source", "unknown")) for index in indices], dtype=object)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), dpi=200)
    scatter_labels(axes[0], coords, labels)
    axes[0].set_title("Severity label")
    cmap = plt.get_cmap("tab10")
    for position, source in enumerate(sorted(set(sources))):
        mask = sources == source
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.58, color=cmap(position % 10), label=source)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_title("Source dataset")
    axes[1].grid(alpha=0.25, linewidth=0.4)
    for label_source, color, marker in (("true", "#111827", "o"), ("pseudo", "#E11D48", "x")):
        mask = label_sources == label_source
        axes[2].scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.55, color=color, marker=marker, label=label_source)
    axes[2].legend(fontsize=8)
    axes[2].set_title("True vs pseudo")
    axes[2].grid(alpha=0.25, linewidth=0.4)
    fig.suptitle("Best Mocha retrained on all Skeleton-6 samples: fused A+B embedding")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    result["tsne"] = coords


def save_fold_tsne(results: dict[str, dict[str, Any]], output: Path, title: str, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=200)
    for ax, (source, result) in zip(axes.flat, results.items(), strict=True):
        coords = run_tsne(result["fused_embeddings"], int(args.seed), float(args.tsne_perplexity))
        scatter_labels(ax, coords, result["labels"])
        adjusted_score = metrics_dict(result["labels"], result["predictions"])["macro_f1"]
        plain_score = metrics_dict(result["labels"], result["raw_predictions"])["macro_f1"]
        ax.set_title(f"{source} (adjusted={adjusted_score:.4f}, plain={plain_score:.4f})")
        result["tsne"] = coords
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_embedding_csv(results: dict[str, dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "cache_index", "label", "prediction", "tsne_x", "tsne_y"])
        for split, result in results.items():
            coords = result.get("tsne", np.full((len(result["labels"]), 2), np.nan))
            for position, index in enumerate(result["indices"]):
                writer.writerow(
                    [split, int(index), int(result["labels"][position]), int(result["predictions"][position]),
                     float(coords[position, 0]), float(coords[position, 1])]
                )


def build_submission(
    template: Path,
    trainall_dir: Path,
    output_dir: Path,
    output_zip: Path,
) -> None:
    package_parent = output_dir / "submission_package"
    package_root = package_parent / "Best_mocha_s6_trainall"
    if package_parent.exists():
        shutil.rmtree(package_parent)
    shutil.copytree(template, package_root, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc"))
    final_dir = package_root / "weights/final_ensemble"
    for filename in ("model_a.pth", "model_b.pth", "mean.npy", "std.npy", "config.json"):
        shutil.copy2(trainall_dir / filename, final_dir / filename)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_parent).as_posix())


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "indices": result["indices"],
        "labels": result["labels"],
        "raw_predictions": result["raw_predictions"],
        "predictions": result["predictions"],
        "model_a_embeddings": result["model_a_embeddings"].astype(np.float16),
        "model_b_embeddings": result["model_b_embeddings"].astype(np.float16),
        "fused_embeddings": result["fused_embeddings"].astype(np.float16),
        "tsne": result.get("tsne"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tsne").mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    motions = torch.as_tensor(payload["motions"])
    lengths = torch.as_tensor(payload["lengths"]).long()
    labels = torch.as_tensor(payload["labels"]).long()
    meta = list(payload["meta"])
    if len(labels) != 4453:
        raise ValueError(f"Expected the complete S6 cache with 4,453 rows, got {len(labels)}")
    sources = np.asarray([source_name(row) for row in meta], dtype=object)
    label_sources = np.asarray([str(row.get("label_source", "unknown")).lower() for row in meta], dtype=object)
    old_mean = torch.from_numpy(np.load(ROOT / "weights/stats/pdgam/mean.npy")).float()
    old_std = torch.from_numpy(np.load(ROOT / "weights/stats/pdgam/std.npy")).float()
    text_anchors = create_text_anchors(args.bge_model, args.output_dir)

    all_indices = np.arange(len(labels), dtype=np.int64)
    trainall_a, trainall_b, trainall_norm, trainall_prior, trainall_counts = train_pair(
        run_name="trainall",
        motions=motions,
        lengths=lengths,
        labels=labels,
        train_indices=all_indices,
        old_mean=old_mean,
        old_std=old_std,
        text_anchors=text_anchors,
        args=args,
        device=device,
        output_dir=args.output_dir,
        seed_offset=0,
    )
    trainall_result = infer_pair(
        model_a=trainall_a,
        model_b=trainall_b,
        normalizer=trainall_norm,
        log_prior=trainall_prior,
        motions=motions,
        lengths=lengths,
        labels=labels,
        indices=all_indices,
        args=args,
        device=device,
    )
    save_trainall_tsne(
        trainall_result, meta, args.output_dir / "tsne/trainall_s6_true_pseudo_tsne.png", args,
    )
    trainall_metrics = {
        "adjusted": metrics_dict(trainall_result["labels"], trainall_result["predictions"]),
        "plain_ensemble": metrics_dict(trainall_result["labels"], trainall_result["raw_predictions"]),
        "class_counts": trainall_counts,
        "true_pseudo_counts": dict(Counter(label_sources.tolist())),
    }

    clean_results: dict[str, dict[str, Any]] = {}
    clean_metrics_adjusted: dict[str, dict[str, float]] = {}
    clean_metrics_plain: dict[str, dict[str, float]] = {}
    for fold, held_source in enumerate(TRUE_SOURCES, start=1):
        test_indices = np.flatnonzero((sources == held_source) & (label_sources == "true"))
        train_indices = np.flatnonzero(~((sources == held_source) & (label_sources == "true")))
        model_a, model_b, normalizer, log_prior, _ = train_pair(
            run_name=f"clean_holdout/{held_source}",
            motions=motions,
            lengths=lengths,
            labels=labels,
            train_indices=train_indices,
            old_mean=old_mean,
            old_std=old_std,
            text_anchors=text_anchors,
            args=args,
            device=device,
            output_dir=args.output_dir,
            seed_offset=fold * 101,
        )
        result = infer_pair(
            model_a=model_a,
            model_b=model_b,
            normalizer=normalizer,
            log_prior=log_prior,
            motions=motions,
            lengths=lengths,
            labels=labels,
            indices=test_indices,
            args=args,
            device=device,
        )
        clean_results[held_source] = result
        clean_metrics_adjusted[held_source] = metrics_dict(result["labels"], result["predictions"])
        clean_metrics_plain[held_source] = metrics_dict(result["labels"], result["raw_predictions"])
        print(
            f"[clean holdout] {held_source}: adjusted={clean_metrics_adjusted[held_source]} "
            f"plain={clean_metrics_plain[held_source]}",
            flush=True,
        )

    save_fold_tsne(
        clean_results,
        args.output_dir / "tsne/clean_holdout_true_tsne.png",
        "Best Mocha S6 fully clean source holdout: fused A+B test embeddings",
        args,
    )
    clean_adjusted_scores = [clean_metrics_adjusted[source]["macro_f1"] for source in TRUE_SOURCES]
    clean_plain_scores = [clean_metrics_plain[source]["macro_f1"] for source in TRUE_SOURCES]
    clean_summary = {
        "adjusted": {
            "by_source": clean_metrics_adjusted,
            "macro_f1_mean": float(np.mean(clean_adjusted_scores)),
            "macro_f1_hmean": harmonic_mean(clean_adjusted_scores),
        },
        "plain_no_logit_adjustment": {
            "by_source": clean_metrics_plain,
            "macro_f1_mean": float(np.mean(clean_plain_scores)),
            "macro_f1_hmean": harmonic_mean(clean_plain_scores),
        },
    }

    pdgam_train_indices = np.flatnonzero((sources == "PD-GaM") & (label_sources == "true"))
    pdgam_a, pdgam_b, pdgam_norm, pdgam_prior, _ = train_pair(
        run_name="pdgam_only",
        motions=motions,
        lengths=lengths,
        labels=labels,
        train_indices=pdgam_train_indices,
        old_mean=old_mean,
        old_std=old_std,
        text_anchors=text_anchors,
        args=args,
        device=device,
        output_dir=args.output_dir,
        seed_offset=909,
    )
    pdgam_results: dict[str, dict[str, Any]] = {}
    pdgam_metrics_adjusted: dict[str, dict[str, float]] = {}
    pdgam_metrics_plain: dict[str, dict[str, float]] = {}
    for target_source in TRUE_SOURCES[1:]:
        test_indices = np.flatnonzero((sources == target_source) & (label_sources == "true"))
        result = infer_pair(
            model_a=pdgam_a,
            model_b=pdgam_b,
            normalizer=pdgam_norm,
            log_prior=pdgam_prior,
            motions=motions,
            lengths=lengths,
            labels=labels,
            indices=test_indices,
            args=args,
            device=device,
        )
        pdgam_results[target_source] = result
        pdgam_metrics_adjusted[target_source] = metrics_dict(result["labels"], result["predictions"])
        pdgam_metrics_plain[target_source] = metrics_dict(result["labels"], result["raw_predictions"])
        print(
            f"[PD-GaM only] {target_source}: adjusted={pdgam_metrics_adjusted[target_source]} "
            f"plain={pdgam_metrics_plain[target_source]}",
            flush=True,
        )

    pooled_labels = np.concatenate([result["labels"] for result in pdgam_results.values()])
    pooled_predictions = np.concatenate([result["predictions"] for result in pdgam_results.values()])
    pooled_raw_predictions = np.concatenate([result["raw_predictions"] for result in pdgam_results.values()])
    pdgam_adjusted_scores = [pdgam_metrics_adjusted[source]["macro_f1"] for source in TRUE_SOURCES[1:]]
    pdgam_plain_scores = [pdgam_metrics_plain[source]["macro_f1"] for source in TRUE_SOURCES[1:]]
    pdgam_summary = {
        "adjusted": {
            "by_source": pdgam_metrics_adjusted,
            "pooled": metrics_dict(pooled_labels, pooled_predictions),
            "macro_f1_mean": float(np.mean(pdgam_adjusted_scores)),
            "macro_f1_hmean": harmonic_mean(pdgam_adjusted_scores),
        },
        "plain_no_logit_adjustment": {
            "by_source": pdgam_metrics_plain,
            "pooled": metrics_dict(pooled_labels, pooled_raw_predictions),
            "macro_f1_mean": float(np.mean(pdgam_plain_scores)),
            "macro_f1_hmean": harmonic_mean(pdgam_plain_scores),
        },
    }
    pdgam_plot_results = dict(pdgam_results)
    # Keep the familiar 2x2 layout and use the fourth panel for all three targets pooled.
    pdgam_plot_results["All three pooled"] = {
        "indices": np.concatenate([result["indices"] for result in pdgam_results.values()]),
        "labels": pooled_labels,
        "raw_predictions": pooled_raw_predictions,
        "predictions": pooled_predictions,
        "fused_embeddings": np.concatenate([result["fused_embeddings"] for result in pdgam_results.values()]),
    }
    save_fold_tsne(
        pdgam_plot_results,
        args.output_dir / "tsne/pdgam_only_three_unseen_true_tsne.png",
        "Best Mocha trained only on PD-GaM true: unseen source test embeddings",
        args,
    )

    save_embedding_csv(clean_results, args.output_dir / "tsne/clean_holdout_tsne_points.csv")
    save_embedding_csv(pdgam_plot_results, args.output_dir / "tsne/pdgam_only_tsne_points.csv")
    torch.save(
        {
            "trainall": serializable_result(trainall_result),
            "clean_holdout": {key: serializable_result(value) for key, value in clean_results.items()},
            "pdgam_only": {key: serializable_result(value) for key, value in pdgam_results.items()},
        },
        args.output_dir / "tsne/embedding_results.pt",
    )

    summary = {
        "method": "Best Mocha two-BiGRU probability ensemble with BGE-small text anchoring and tau=0.4 logit adjustment",
        "dataset": "CARE-PD-Skeleton-6 true+pseudo",
        "trainall": trainall_metrics,
        "clean_holdout": clean_summary,
        "pdgam_only": pdgam_summary,
        "settings": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_contrast": float(args.lambda_contrast),
            "temperature": float(args.temperature),
            "tau": float(args.tau),
            "seed": int(args.seed),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    build_submission(args.template, args.output_dir / "trainall", args.output_dir, args.submission_zip)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Submission: {args.submission_zip}", flush=True)


if __name__ == "__main__":
    main()
