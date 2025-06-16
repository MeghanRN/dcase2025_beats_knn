#!/usr/bin/env python
"""
GPU-accelerated inference for DCASE-2025 Task 2 (eval set only).

Produces:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
under your csv_out_dir (configs/default.yaml).
"""

import argparse, sys, glob
from pathlib import Path

import numpy as np
import torch, torchaudio
from tqdm.auto import tqdm

from src.utils.file_utils      import load_config
from src.models.beats_backbone import BEATsBackbone

def main():
    # 1) Argparse & config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args   = p.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Paths
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) Load backbone & raw memory-bank (CPU)
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    ckpt     = torch.load(bank_dir/"memory_bank.pt", map_location="cpu")
    raw_mem  = ckpt["memory"]  # list of [1×D] tensors

    # 4) Build GPU mem_bank of shape (N_train, D)
    mem_bank = torch.stack([x.squeeze(0) for x in raw_mem], dim=0).to(device)
    N, D = mem_bank.shape
    K   = cfg["model"]["k"]

    # 5) Compute decision threshold on GPU in chunks
    print("▶ Computing threshold distances on GPU…")
    mem_dists = []
    chunk     = 1024  # adjust to fit your GPU
    for i in tqdm(range(0, N, chunk),
                  desc="  threshold chunks",
                  unit="chunk",
                  file=sys.stdout,
                  dynamic_ncols=True):
        sub = mem_bank[i : i+chunk]           # (chunk, D)
        # full pairwise distances (chunk, N)
        d = torch.cdist(sub, mem_bank)        # GPU
        # ignore self-dist: set diagonal in each row-block to large value
        for j in range(d.size(0)):
            global_idx = i + j
            if global_idx < N:
                d[j, global_idx] = float("inf")
        # take k smallest per row
        topk = torch.topk(d, k=K, dim=1, largest=False).values  # (chunk, K)
        mem_dists.append(topk.mean(dim=1).cpu().numpy())

    mem_dists = np.concatenate(mem_dists, axis=0)  # (N,)
    pct       = cfg.get("threshold", {}).get("percentile", 90)
    threshold = float(np.percentile(mem_dists, pct))
    print(f"▶ Decision threshold @ {pct}th percentile = {threshold:.6f}")

    # 6) Gather eval_data test WAVs
    root    = cfg["data"]["root"]
    pattern = f"{root}/eval_data/raw/*/test/**/*.wav"
    wavs    = sorted(glob.glob(pattern, recursive=True))
    print(f"▶ Found {len(wavs)} eval clips under: {pattern}")
    if not wavs:
        print("⚠️  No eval_data test files found! Did you run download_task2_data.sh?")
        sys.exit(1)

    # 7) Inference loop with live GPU-knn scoring
    writers = {}
    for path in tqdm(wavs,
                     desc="Scoring eval clips",
                     unit="clip",
                     file=sys.stdout,
                     leave=True,
                     dynamic_ncols=True):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device); sr = int(sr)

        # backbone→embedding (1, D)
        feat = backbone(wav, sr)

        # GPU k‐NN scoring
        d    = torch.cdist(feat, mem_bank)            # (1, N)
        topk = torch.topk(d, k=K, dim=1, largest=False).values  # (1, K)
        score = float(topk.mean().item())
        decision = 1 if score > threshold else 0

        p       = Path(path)
        machine = p.parent.parent.name      # e.g. CoffeeGrinder
        section = p.stem.split("_")[1]         # take "section_XX_####" → ["section","XX","####"]
        tag     = f"{machine}_section_{section}"

        # open CSVs on first use
        if tag not in writers:
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (asc.open("w"), dec.open("w"))

        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # 8) Close filehandles
    for asc_fp, dec_fp in writers.values():
        asc_fp.close()
        dec_fp.close()

    print(f"✅ All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
