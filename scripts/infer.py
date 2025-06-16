#!/usr/bin/env python
"""
Final inference script for DCASE-2025 Task-2 (eval set only), fully GPU-accelerated KNN.

Creates, for each evaluation machine & section:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
under the CSV output directory specified in configs/default.yaml.
"""

import argparse
import sys
import glob
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.utils.file_utils      import load_config
from src.models.beats_backbone import BEATsBackbone


def main():
    # 1) parse arguments & load config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args   = p.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) prepare output directories
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) load backbone & memory bank
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    ckpt     = torch.load(bank_dir / "memory_bank.pt", map_location=device)
    mem_bank = torch.stack(ckpt["memory"], dim=0).to(device)  # (N_train, D)
    K         = cfg["model"]["k"]

    # 4) compute decision threshold on GPU
    pct = cfg.get("threshold", {}).get("percentile", 90)
    mem_dists = []
    chunk = 2048
    for i in tqdm(
        range(0, mem_bank.size(0), chunk),
        desc="Computing threshold dists",
        unit="chunk",
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        sub = mem_bank[i : i + chunk]            # (chunk, D)
        d   = torch.cdist(sub, mem_bank)         # (chunk, N_train)
        topk = torch.topk(d, k=K, dim=1, largest=False).values  # (chunk, K)
        mem_dists.append(topk.mean(dim=1).cpu().numpy())
    mem_dists = np.concatenate(mem_dists, axis=0)
    threshold = float(np.percentile(mem_dists, pct))
    print(f"Decision threshold @ {pct}-percentile: {threshold:.6f}")

    # 5) glob eval_data test WAVs
    root    = Path(cfg["data"]["root"]) / "eval_data" / "raw"
    pattern = root / "*" / "test" / "**" / "*.wav"
    wavs    = sorted(glob.glob(str(pattern), recursive=True))
    print(f"▶ Found {len(wavs)} eval clips under: {pattern}")
    if not wavs:
        print("⚠️  No eval_data test files found! Did you run download_task2_data.sh?")
        return

    # 6) run inference with live tqdm bar
    writers: dict[str, tuple[any, any]] = {}
    for path in tqdm(
        wavs,
        desc="Scoring eval clips",
        unit="clip",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    ):
        # load & mono-mix
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        sr  = int(sr)

        # extract embedding
        feat = backbone(wav, sr)                # (1, D) on GPU

        # GPU KNN scoring
        d    = torch.cdist(feat, mem_bank)      # (1, N_train)
        topk = torch.topk(d, k=K, dim=1, largest=False).values  # (1, K)
        score = float(topk.mean().item())

        decision = 1 if score > threshold else 0

        p = Path(path)
        # machine: eval_data/raw/<machine>/test/…
        machine = p.parent.parent.name
        # section: folder "section_XX"
        section = p.parent.name.split("_")[1]
        tag     = f"{machine}_section_{section}"

        # open CSVs on first encounter
        if tag not in writers:
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (asc.open("w"), dec.open("w"))

        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # 7) close all file handles
    for asc_fp, dec_fp in writers.values():
        asc_fp.close()
        dec_fp.close()

    print(f"✅ All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
