#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_best_mocha_s6_full_eval import GRUClassifier, GRUTextAnchored, Normalizer, make_loader


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--model-a", type=Path, required=True)
    p.add_argument("--model-b", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--label-subset", choices=("true", "all"), default="true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def f1_scores(conf):
    values = []
    for k in range(4):
        tp, fp, fn = conf[k, k], conf[:, k].sum() - conf[k, k], conf[k, :].sum() - conf[k, k]
        values.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return values


def main():
    a = args()
    metadata_path = a.dataset / "metadata.csv"
    if not metadata_path.is_file() and (a.dataset / "CARE-PD-Skeleton-6/metadata.csv").is_file():
        metadata_path = a.dataset / "CARE-PD-Skeleton-6/metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found below {a.dataset}")
    external = pd.read_csv(metadata_path)
    external_ids = set(external["sample_id"].astype(str))

    payloads = [torch.load(a.cache_dir / f"{tag}_motion_smpl_recon_cache.pt", map_location="cpu")
                for tag in ("pdgam", "bmclab", "tsdupd", "3dgait")]
    motions = torch.cat([torch.as_tensor(x["motions"]) for x in payloads])
    lengths = torch.cat([torch.as_tensor(x["lengths"]).long() for x in payloads])
    labels = torch.cat([torch.as_tensor(x["labels"]).long() for x in payloads])
    meta = sum([list(x["meta"]) for x in payloads], [])
    missing = [str(row["sample_id"]) for row in meta if str(row["sample_id"]) not in external_ids]
    if missing:
        raise ValueError(f"Dataset does not match bundled training/eval cache; missing sample IDs: {missing[:5]}")
    keep = np.asarray([a.label_subset == "all" or str(row.get("label_source", "")).lower() == "true" for row in meta])
    indices = np.flatnonzero(keep)

    root = Path(__file__).resolve().parents[1]
    normalizer = Normalizer(
        torch.from_numpy(np.load(root / "weights/stats/pdgam/mean.npy")).float(),
        torch.from_numpy(np.load(root / "weights/stats/pdgam/std.npy")).float(),
        torch.from_numpy(np.load(a.template / "weights/final_ensemble/mean.npy")).float(),
        torch.from_numpy(np.load(a.template / "weights/final_ensemble/std.npy")).float(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, model_b = GRUClassifier().to(device), GRUTextAnchored().to(device)
    model_a.load_state_dict(torch.load(a.model_a, map_location="cpu"), strict=True)
    model_b.load_state_dict(torch.load(a.model_b, map_location="cpu"), strict=True)
    model_a.eval(); model_b.eval()
    predictions, targets = [], []
    loader = make_loader(motions, lengths, labels, indices, a.batch_size, False, 42)
    with torch.inference_mode():
        for motion, lens, target, _ in loader:
            x = normalizer.transform(motion, lens, device)
            logits = (model_a(x, lens) + model_b(x, lens)) / 2
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(target.tolist())
    conf = np.zeros((4, 4), dtype=np.int64)
    for y, pred in zip(targets, predictions): conf[y, pred] += 1
    f1 = f1_scores(conf)
    result = {"dataset": str(metadata_path), "label_subset": a.label_subset, "samples": len(targets),
              "accuracy": float(np.trace(conf) / max(conf.sum(), 1)), "macro_f1": float(np.mean(f1)),
              "per_class_f1": f1, "confusion_matrix": conf.tolist()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved={a.output}")


if __name__ == "__main__": main()

