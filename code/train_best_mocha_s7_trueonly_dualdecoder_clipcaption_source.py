#!/usr/bin/env python3
"""Weakly fine-tune Best-MoCHA A/B on one S7 source with CLIP captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from train_best_mocha_s6_dualdecoder_recon_supcon import (
    MotionSequenceDecoder,
    masked_reconstruction_loss,
)
from train_best_mocha_s6_full_eval import (
    GRUClassifier,
    GRUTextAnchored,
    Normalizer,
    make_loader,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "artifacts/validation/best_mocha_submission/Best_mocha_submission"
DEFAULT_CAPTIONS = Path("/data/data/CARE-PD-Skeleton-6/llm_captions")
DEFAULT_CLIP = ROOT / "artifacts/models/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--caption-dir", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-subset",
        choices=("true", "all"),
        default="true",
        help="Use only true labels or every true+pseudo sample in the source cache.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr-backbone", type=float, default=2e-6)
    parser.add_argument("--lr-head", type=float, default=7e-5)
    parser.add_argument("--lr-aux", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--caption-weight", type=float, default=1.0)
    parser.add_argument("--recon-a-weight", type=float, default=0.05)
    parser.add_argument("--recon-b-weight", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.0,
        help="Label-based supervised contrastive loss weight applied independently to both branches.",
    )
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=4,
        default=None,
        metavar=("W0", "W1", "W2", "W3"),
        help="Optional class weights applied to both Model A and Model B cross-entropy losses.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_name(row: dict[str, Any]) -> str:
    return str(row.get("source", "")).removesuffix("_canonical.pkl").removesuffix(".pkl")


def load_sample_captions(
    metadata: Sequence[dict[str, Any]], caption_dir: Path
) -> tuple[list[str], torch.Tensor]:
    caption_maps = {
        path.name.removesuffix("_llm_captions.json"): json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(caption_dir.glob("*_llm_captions.json"))
    }
    if not caption_maps:
        raise FileNotFoundError(f"No caption JSON files found in {caption_dir}")
    captions: list[str] = []
    available = torch.zeros(len(metadata), dtype=torch.bool)
    for index, row in enumerate(metadata):
        key = f"{row.get('subject_id')}::{row.get('walk_id')}"
        caption = caption_maps.get(source_name(row), {}).get(key)
        captions.append(str(caption) if caption is not None else "")
        available[index] = bool(caption)
    return captions, available


def caption_signature(captions: Sequence[str], available: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for caption, present in zip(captions, available.tolist()):
        digest.update(b"1" if present else b"0")
        digest.update(caption.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def encode_clip_captions(
    captions: Sequence[str],
    available: torch.Tensor,
    model_path: Path,
    cache_path: Path,
    device: torch.device,
) -> torch.Tensor:
    signature = caption_signature(captions, available)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        embeddings = torch.as_tensor(payload["embeddings"]).float()
        if payload.get("signature") == signature and embeddings.shape == (len(captions), 512):
            print(f"Loaded cached CLIP captions from {cache_path}", flush=True)
            return embeddings

    tokenizer = CLIPTokenizerFast.from_pretrained(model_path, local_files_only=True)
    encoder = CLIPTextModelWithProjection.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    present_indices = torch.where(available)[0].tolist()
    present_captions = [captions[index] for index in present_indices]
    encoded_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(present_captions), 128):
            batch = tokenizer(
                present_captions[start : start + 128],
                padding=True,
                truncation=True,
                max_length=int(tokenizer.model_max_length),
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            encoded_chunks.append(F.normalize(encoder(**batch).text_embeds.float(), dim=1).cpu())
    encoded = torch.cat(encoded_chunks, dim=0)
    embeddings = torch.zeros((len(captions), 512), dtype=torch.float32)
    embeddings[present_indices] = encoded
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings.half(),
            "available": available,
            "signature": signature,
            "model": str(model_path),
        },
        cache_path,
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved CLIP captions to {cache_path}", flush=True)
    return embeddings


def paired_contrastive_loss(
    motion_features: torch.Tensor, text_features: torch.Tensor, temperature: float
) -> torch.Tensor:
    if motion_features.shape[0] < 2:
        return motion_features.sum() * 0.0
    motion_features = F.normalize(motion_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    logits = motion_features @ text_features.T / float(temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2.0


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Label-based SupCon over one embedding view per sample."""
    if features.shape[0] <= 1:
        return features.sum() * 0.0
    features = F.normalize(features.float(), dim=1)
    labels = labels.view(-1, 1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels.eq(labels.T) & ~self_mask
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return features.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~self_mask).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_prob = (
        (log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    return -mean_positive_log_prob[valid].mean()


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"source={args.source_name} device={device}", flush=True)

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    motions = torch.as_tensor(payload["motions"])
    lengths = torch.as_tensor(payload["lengths"]).long()
    labels = torch.as_tensor(payload["labels"]).long()
    metadata = list(payload["meta"])
    if not (len(motions) == len(lengths) == len(labels) == len(metadata)):
        raise ValueError("Cache tensors and metadata have inconsistent lengths")
    label_sources = [str(row.get("label_source", "")).lower() for row in metadata]
    true_mask = torch.as_tensor(
        [label_source == "true" for label_source in label_sources],
        dtype=torch.bool,
    )
    pseudo_mask = torch.as_tensor(
        [label_source == "pseudo" for label_source in label_sources], dtype=torch.bool
    )
    selected_mask = (
        true_mask if args.label_subset == "true" else torch.ones(len(metadata), dtype=torch.bool)
    )
    indices = np.flatnonzero(selected_mask.numpy()).astype(np.int64)
    if not len(indices):
        raise ValueError(f"No selected samples for source {args.source_name}")
    true_counts = torch.bincount(labels[true_mask], minlength=4)
    pseudo_counts = torch.bincount(labels[pseudo_mask], minlength=4)
    selected_counts = torch.bincount(labels[selected_mask], minlength=4)

    captions, caption_available = load_sample_captions(metadata, args.caption_dir)
    matched = int((caption_available & selected_mask).sum())
    clip_embeddings = encode_clip_captions(
        captions,
        caption_available,
        args.clip_model,
        args.output_dir / "clip_caption_embeddings.pt",
        device,
    )
    print(
        f"label_subset={args.label_subset} samples={len(indices)} "
        f"selected_counts={selected_counts.tolist()} true_counts={true_counts.tolist()} "
        f"pseudo_counts={pseudo_counts.tolist()} "
        f"captions={matched} missing={len(indices) - matched}",
        flush=True,
    )

    old_mean = torch.from_numpy(np.load(ROOT / "weights/stats/pdgam/mean.npy")).float()
    old_std = torch.from_numpy(np.load(ROOT / "weights/stats/pdgam/std.npy")).float()
    base_weights = args.template / "weights/final_ensemble"
    pretrained_mean = torch.from_numpy(np.load(base_weights / "mean.npy")).float()
    pretrained_std = torch.from_numpy(np.load(base_weights / "std.npy")).float()
    normalizer = Normalizer(old_mean, old_std, pretrained_mean, pretrained_std)

    model_a = GRUClassifier().to(device)
    model_b = GRUTextAnchored().to(device)
    model_a.load_state_dict(
        torch.load(base_weights / "model_a.pth", map_location="cpu", weights_only=False),
        strict=True,
    )
    model_b.load_state_dict(
        torch.load(base_weights / "model_b.pth", map_location="cpu", weights_only=False),
        strict=True,
    )
    decoder_a = MotionSequenceDecoder().to(device)
    decoder_b = MotionSequenceDecoder().to(device)
    clip_adapter_b = nn.Linear(384, 512, bias=False).to(device)

    backbone_parameters = list(model_a.gru.parameters()) + list(model_b.gru.parameters())
    head_parameters = (
        list(model_a.classifier.parameters())
        + list(model_b.classifier.parameters())
        + list(model_b.text_proj.parameters())
    )
    core_parameters = backbone_parameters + head_parameters
    auxiliary_parameters = (
        list(decoder_a.parameters())
        + list(decoder_b.parameters())
        + list(clip_adapter_b.parameters())
    )
    if not all(parameter.requires_grad for parameter in core_parameters):
        raise RuntimeError("Both pretrained Best-MoCHA branches must be fully trainable")
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(args.lr_backbone)},
            {"params": head_parameters, "lr": float(args.lr_head)},
            {"params": auxiliary_parameters, "lr": float(args.lr_aux)},
        ],
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.epochs), 1)
    )
    loader = make_loader(
        motions, lengths, labels, indices, int(args.batch_size), True, int(args.seed)
    )
    class_weights = None
    if args.class_weights is not None:
        class_weights = torch.as_tensor(args.class_weights, dtype=torch.float32, device=device)
        if not torch.isfinite(class_weights).all() or bool((class_weights <= 0).any()):
            raise ValueError("--class-weights must contain four finite positive values")
        print(f"CE class weights={class_weights.detach().cpu().tolist()}", flush=True)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model_a.train()
        model_b.train()
        decoder_a.train()
        decoder_b.train()
        clip_adapter_b.train()
        totals = np.zeros(8, dtype=np.float64)
        seen = 0
        for batch_motion, batch_lengths, batch_labels, batch_indices in loader:
            batch_labels = batch_labels.to(device, non_blocking=True)
            motion = normalizer.transform(batch_motion, batch_lengths, device)
            optimizer.zero_grad(set_to_none=True)

            embedding_a = model_a.encode(motion, batch_lengths)
            embedding_b = model_b.encode(motion, batch_lengths)
            ce_a = F.cross_entropy(
                model_a.classifier(embedding_a), batch_labels, weight=class_weights
            )
            ce_b = F.cross_entropy(
                model_b.classifier(embedding_b), batch_labels, weight=class_weights
            )
            supcon_a = supervised_contrastive_loss(
                embedding_a, batch_labels, float(args.supcon_temperature)
            )
            supcon_b = supervised_contrastive_loss(
                embedding_b, batch_labels, float(args.supcon_temperature)
            )
            recon_a = masked_reconstruction_loss(
                decoder_a(embedding_a, motion.shape[1]), motion, batch_lengths
            )
            recon_b = masked_reconstruction_loss(
                decoder_b(embedding_b, motion.shape[1]), motion, batch_lengths
            )
            caption_mask = caption_available[batch_indices]
            if int(caption_mask.sum()) >= 2:
                motion_text = clip_adapter_b(
                    model_b.text_proj(embedding_b[caption_mask.to(device)])
                )
                caption_text = clip_embeddings[batch_indices[caption_mask]].to(
                    device=device, dtype=torch.float32
                )
                caption_loss = paired_contrastive_loss(
                    motion_text, caption_text, float(args.temperature)
                )
            else:
                caption_loss = embedding_b.sum() * 0.0
            loss = (
                ce_a
                + ce_b
                + float(args.supcon_weight) * (supcon_a + supcon_b)
                + float(args.caption_weight) * caption_loss
                + float(args.recon_a_weight) * recon_a
                + float(args.recon_b_weight) * recon_b
            )
            loss.backward()
            nn.utils.clip_grad_norm_(core_parameters + auxiliary_parameters, 5.0)
            optimizer.step()

            batch_size = int(batch_labels.shape[0])
            seen += batch_size
            totals += np.asarray(
                [
                    loss.item(),
                    ce_a.item(),
                    ce_b.item(),
                    supcon_a.item(),
                    supcon_b.item(),
                    caption_loss.item(),
                    recon_a.item(),
                    recon_b.item(),
                ]
            ) * batch_size
        scheduler.step()
        average = totals / max(seen, 1)
        row = {
            "epoch": epoch,
            "loss": float(average[0]),
            "ce_a": float(average[1]),
            "ce_b": float(average[2]),
            "supcon_a": float(average[3]),
            "supcon_b": float(average[4]),
            "clip_caption_contrast": float(average[5]),
            "recon_a": float(average[6]),
            "recon_b": float(average[7]),
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "lr_aux": float(optimizer.param_groups[2]["lr"]),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(args.epochs):
            print(
                f"[{args.source_name}] epoch={epoch:02d}/{args.epochs} "
                f"loss={row['loss']:.4f} ce=({row['ce_a']:.4f},{row['ce_b']:.4f}) "
                f"supcon=({row['supcon_a']:.4f},{row['supcon_b']:.4f}) "
                f"clip={row['clip_caption_contrast']:.4f} "
                f"recon=({row['recon_a']:.4f},{row['recon_b']:.4f})",
                flush=True,
            )

    torch.save(model_a.state_dict(), args.output_dir / "model_a.pth")
    torch.save(model_b.state_dict(), args.output_dir / "model_b.pth")
    torch.save(decoder_a.state_dict(), args.output_dir / "decoder_a.pth")
    torch.save(decoder_b.state_dict(), args.output_dir / "decoder_b.pth")
    torch.save(clip_adapter_b.state_dict(), args.output_dir / "clip_adapter_b.pth")
    np.save(args.output_dir / "mean.npy", pretrained_mean.numpy())
    np.save(args.output_dir / "std.npy", pretrained_std.numpy())
    metrics = {
        "source_name": args.source_name,
        "cache": str(args.cache),
        "initialized_from": str(base_weights),
        "all_model_a_parameters_trainable": True,
        "all_model_b_parameters_trainable": True,
        "true_only": args.label_subset == "true",
        "label_subset": args.label_subset,
        "samples": len(indices),
        "selected_class_counts": [int(value) for value in selected_counts],
        "true_class_counts": [int(value) for value in true_counts],
        "pseudo_class_counts": [int(value) for value in pseudo_counts],
        "caption_source": str(args.caption_dir),
        "caption_encoder": str(args.clip_model),
        "clip_frozen": True,
        "caption_matched": matched,
        "caption_missing": len(indices) - matched,
        "loss_weights": {
            "ce_a": 1.0,
            "ce_b": 1.0,
            "ce_class_weights": (
                [float(value) for value in args.class_weights]
                if args.class_weights is not None
                else [1.0, 1.0, 1.0, 1.0]
            ),
            "supcon_a": float(args.supcon_weight),
            "supcon_b": float(args.supcon_weight),
            "clip_caption_contrast_b": float(args.caption_weight),
            "reconstruction_a": float(args.recon_a_weight),
            "reconstruction_b": float(args.recon_b_weight),
        },
        "contrast_temperature": float(args.temperature),
        "supcon_temperature": float(args.supcon_temperature),
        "epochs": int(args.epochs),
        "lr_backbone": float(args.lr_backbone),
        "lr_head": float(args.lr_head),
        "lr_aux": float(args.lr_aux),
        "seed": int(args.seed),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved={args.output_dir}", flush=True)


if __name__ == "__main__":
    train(parse_args())
