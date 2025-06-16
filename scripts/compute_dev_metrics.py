#!/usr/bin/env python
"""
Compute AUC_source, AUC_target, and pAUC@FPR0.1 on dev_data test set for each machine.
Usage:
  python scripts/compute_dev_metrics.py --config configs/default.yaml
"""

import argparse, glob, os
from pathlib import Path

import numpy as np
import torch, torchaudio
from sklearn.metrics import roc_auc_score

from src.utils.file_utils     import load_config
from src.models.beats_backbone import BEATsBackbone
from src.models.detector      import KNNDetector


def compute_metrics(cfg_path: str):
    # 1) load config, set up device
    cfg    = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) load backbone & k-NN bank
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    bank_ckpt = torch.load(
        Path(cfg["logging"]["bank_out"]) / "memory_bank.pt",
        map_location=device
    )
    detector = KNNDetector(k=cfg["model"]["k"])
    detector.fit(bank_ckpt["memory"])

    # 3) locate dev_data test files
    root = Path(cfg["data"]["root"]) / "dev_data" / "raw"
    machines = sorted([d.name for d in root.iterdir() if d.is_dir()])

    results = {}

    for m in machines:
        # gather all test clips for this machine
        pattern = str(root / m / "test" / "**" / "*.wav")
        files = glob.glob(pattern, recursive=True)

        y_src, s_src = [], []
        y_tgt, s_tgt = [], []

        for f in files:
            # ground truth: anomaly if filename contains 'anomaly'
            label = 1 if "anomaly" in os.path.basename(f) else 0

            # run inference
            wav, sr = torchaudio.load(f)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            feat  = backbone(wav.to(device), sr)
            score = float(detector.score(feat.cpu()))

            if "_source_test_" in f:
                y_src.append(label)
                s_src.append(score)
            elif "_target_test_" in f:
                y_tgt.append(label)
                s_tgt.append(score)

        # sanity check
        if not y_src or not y_tgt:
            raise RuntimeError(f"No source/target clips found for {m} under {pattern}")

        # compute metrics (as percents)
        auc_source = roc_auc_score(y_src, s_src) * 100
        auc_target = roc_auc_score(y_tgt, s_tgt) * 100
        # partial AUC at FPR<=0.1, normalized to [0,1], then *100
        pauc = roc_auc_score(y_tgt, s_tgt, max_fpr=0.1) * 100

        results[m] = (auc_source, auc_target, pauc)

    # 4) print YAML snippet
    print("results:")
    print("  development_dataset:")
    for m, (asrc, atgt, p) in results.items():
        print(f"    {m}:")
        print(f"      auc_source: {asrc:.2f}")
        print(f"      auc_target: {atgt:.2f}")
        print(f"      pauc: {p:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to your inference/training config"
    )
    args = ap.parse_args()
    compute_metrics(args.config)
