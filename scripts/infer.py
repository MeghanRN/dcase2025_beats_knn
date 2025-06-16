#!/usr/bin/env python
"""
Final inference script for DCASE-2025 Task-2 (eval set only).

Creates:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
in results/csv (as per configs/default.yaml).
"""

import argparse, sys, glob
from pathlib import Path

import numpy as np
import torch, torchaudio
from tqdm.auto import tqdm

from src.utils.file_utils     import load_config
from src.models.beats_backbone import BEATsBackbone
from src.models.detector      import KNNDetector


def main():
    # 1) parse arguments & load config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) prepare directories
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) load backbone & memory bank
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    ckpt     = torch.load(bank_dir/"memory_bank.pt", map_location=device)

    detector = KNNDetector(k=cfg["model"]["k"])
    detector.fit(ckpt["memory"])

    # 4) compute decision threshold
    mem_dists = [float(detector.score(x)) for x in ckpt["memory"]]
    pct       = cfg.get("threshold", {}).get("percentile", 90)
    threshold = float(np.percentile(mem_dists, pct))
    print(f"Decision threshold @ {pct}-percentile: {threshold:.6f}")

    # 5) find eval_data test WAVs
    root = Path(cfg["data"]["root"]) / "eval_data" / "raw"
    pattern = root / "*" / "test" / "**" / "*.wav"
    wavs = sorted(glob.glob(str(pattern), recursive=True))
    print(f"▶ Found {len(wavs)} eval clips under: {pattern}")

    if not wavs:
        print("⚠️  No eval_data test files found! Did you run download_task2_data.sh?")
        return

    # 6) run inference with a live progress bar
    writers: dict[str, tuple] = {}

    for path in tqdm(
        wavs,
        desc="Scoring eval clips",
        unit="clip",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    ):
        # load & preprocess
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        sr  = int(sr)

        # forward & score
        feat     = backbone(wav, sr)
        score    = float(detector.score(feat.cpu()))
        decision = 1 if score > threshold else 0

        p       = Path(path)
        machine = p.parent.parent.name          # eval_data/raw/<machine>/test/...
        section = p.parent.name.split("_")[1]   # "section_XX" → "XX"
        tag     = f"{machine}_section_{section}"

        # open CSVs on first write
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
