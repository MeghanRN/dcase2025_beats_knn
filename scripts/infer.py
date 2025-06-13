#!/usr/bin/env python
"""
Inference for DCASE-2025 Task-2
—————————————
Reads the memory-bank created by `train_knn.py`, computes k-NN
anomaly scores for every *test* clip, and writes one CSV per
machine/section in the official DCASE format.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.models.detector import KNNDetector


# ──────────────────────────────────────────────────────────────
def collate(batch):
    """Keep list of tuples unchanged (variable-length audio)."""
    return batch


def main() -> None:
    # ── CLI -------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve legacy vs. new keys ------------------------------------
    bank_dir = cfg.get("output_dir", cfg["logging"]["bank_out"])
    csv_out  = cfg.get("csv_out_dir", cfg["logging"]["csv_out_dir"])

    # ── dataset & loader ------------------------------------------------
    test_ds = DCASETask2Dataset(cfg["data"]["root"], split="test")
    loader  = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # ── backbone & k-NN detector ---------------------------------------
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()

    ckpt = torch.load(Path(bank_dir) / "memory_bank.pt", map_location=device)
    detector = KNNDetector(k=cfg["model"]["k"])
    detector.fit(ckpt["memory"])         # list[Tensor]

    # ── CSV writers per machine/section --------------------------------
    writers = {}

    for batch in tqdm(loader, desc="Scoring clips"):
        wav, sr, path = batch[0]
        wav = wav.to(device)
        sr  = int(sr) if torch.is_tensor(sr) else sr

        feat  = backbone(wav, sr)
        score = detector.score(feat).item()

        rel = Path(path).relative_to(cfg["data"]["root"])
        # e.g.  dev_data/raw/fan/test/section_00_source_test_anomaly_012.wav
        parts = rel.parts
        machine   = parts[3]      # fan
        section   = parts[4].split("_")[1]  # 00
        csv_name  = f"anomaly_score_{machine}_section_{section}.csv"

        if csv_name not in writers:
            out_dir = Path(csv_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            fp = open(out_dir / csv_name, "w", newline="")
            writers[csv_name] = (fp, csv.writer(fp))
            writers[csv_name][1].writerow(["path", "score"])

        writers[csv_name][1].writerow([path, f"{score:.6f}"])

    # ── close all files -------------------------------------------------
    for fp, _ in writers.values():
        fp.close()

    print(f"✅  CSV files written to {csv_out}/")


if __name__ == "__main__":
    main()
