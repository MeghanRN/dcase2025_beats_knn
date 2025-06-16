#!/usr/bin/env python
"""
Final inference script for DCASE-2025 Task-2 (eval set only),
with GPU-accelerated k-NN scoring and CPU-based threshold compute.

Creates:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
under your csv_out_dir (as set in configs/default.yaml).
"""

import argparse
import sys
import glob
from pathlib import Path

import numpy as np
import torch, torchaudio
from tqdm.auto import tqdm

from src.utils.file_utils      import load_config
from src.models.beats_backbone import BEATsBackbone
from src.models.detector       import KNNDetector


def main():
    # 1) parse args + load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) prepare dirs
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) load backbone & raw memory bank (on CPU)
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    ckpt     = torch.load(bank_dir/"memory_bank.pt", map_location="cpu")
    raw_mem  = ckpt["memory"]   # list of (1×D) Tensors

    # 4) compute threshold on CPU via sklearn-kNN
    detector_cpu = KNNDetector(k=cfg["model"]["k"])
    detector_cpu.fit(raw_mem)
    mem_dists = [float(detector_cpu.score(x)) for x in raw_mem]
    pct       = cfg.get("threshold", {}).get("percentile", 90)
    threshold = float(np.percentile(mem_dists, pct))
    print(f"Decision threshold @ {pct}-percentile: {threshold:.6f}")

    # 5) build GPU memory bank for fast scoring
    mem_bank = torch.stack([x.squeeze(0) for x in raw_mem], dim=0).to(device)  # (N_train, D)
    K        = cfg["model"]["k"]

    # 6) glob eval_data test WAVs (fixed syntax!)
    root    = cfg["data"]["root"]
    pattern = f"{root}/eval_data/raw/*/test/**/*.wav"
    wavs    = sorted(glob.glob(pattern, recursive=True))
    print(f"▶ Found {len(wavs)} eval clips under: {pattern}")
    if not wavs:
        print("⚠️  No eval_data test files found! Did you run download_task2_data.sh?")
        sys.exit(1)

    # 7) inference loop w/ live tqdm bar
    writers = {}
    for path in tqdm(
        wavs,
        desc="Scoring eval clips",
        unit="clip",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    ):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device); sr = int(sr)

        # get embedding
        feat = backbone(wav, sr)                 # (1, D) on GPU

        # GPU k-NN scoring
        d    = torch.cdist(feat, mem_bank)       # (1, N_train)
        topk = torch.topk(d, k=K, dim=1, largest=False).values  # (1, K)
        score = float(topk.mean().item())

        decision = 1 if score > threshold else 0

        p       = Path(path)
        machine = p.parent.parent.name                 # e.g. CoffeeGrinder
        section = p.parent.name.split("_")[1]          # e.g. "00"
        tag     = f"{machine}_section_{section}"

        if tag not in writers:
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (asc.open("w"), dec.open("w"))

        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # 8) close all file handles
    for asc_fp, dec_fp in writers.values():
        asc_fp.close(); dec_fp.close()

    print(f"✅ All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
