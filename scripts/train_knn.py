#!/usr/bin/env python
"""
Build a k-NN memory bank for DCASE-2025 Task 2
using HuBERT-Base embeddings.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.models.detector import KNNDetector


# ──────────────────────────────────────────────────────────────
def collate(batch):
    """Keep list of (wav, sr, path) tuples unchanged."""
    return batch


def main() -> None:
    # ── parse CLI -------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset & loader -----------------------------------------------
    root = cfg["data"]["root"]
    dataset = DCASETask2Dataset(root, split="train")

    loader = DataLoader(
        dataset,
        batch_size=1,                     # leave at 1; HuBERT is heavy on CPU
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # ── model & detector -----------------------------------------------
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    detector = KNNDetector(k=cfg["model"]["k"])

    feats, paths = [], []

    # ── forward pass ----------------------------------------------------
    for batch in tqdm(loader, desc="Extracting embeddings"):
        waveform, sr, path = batch[0]          # unpack 1-item list
        waveform = waveform.to(device)
        sr = int(sr) if torch.is_tensor(sr) else sr

        feat = backbone(waveform, sr)          # (1, D)
        feats.append(feat.cpu())
        paths.append(path)

    # ── fit k-NN memory bank -------------------------------------------
    detector.fit(feats)

    out_dir = Path(cfg["logging"]["bank_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"memory": feats, "paths": paths}, out_dir / "memory_bank.pt")
    print(f"✅  Memory bank saved → {out_dir/'memory_bank.pt'}")


if __name__ == "__main__":
    main()
