#!/usr/bin/env python
"""
Debug version of inference for DCASE-2025 Task-2 eval set.
Prints explicit status messages so you can see exactly what's happening.
"""

import argparse, sys, glob
from pathlib import Path

import numpy as np
import torch, torchaudio
from tqdm.auto import tqdm

from src.utils.file_utils      import load_config
from src.models.beats_backbone import BEATsBackbone
from src.models.detector       import KNNDetector


def main():
    print("▶ Starting infer.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args   = parser.parse_args()

    # 1) load config & device
    print(f"▶ Loading config: {args.config}")
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    # 2) output dirs
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    print(f"▶ Memory bank folder: {bank_dir.resolve()}")
    print(f"▶ CSV output folder:  {csv_dir.resolve()}")
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) load backbone & bank
    print("▶ Loading fine-tuned backbone & memory bank…")
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    print("  ✔ Backbone loaded")
    ckpt = torch.load(bank_dir/"memory_bank.pt", map_location="cpu")
    raw_mem = ckpt["memory"]
    print(f"  ✔ Loaded memory_bank.pt with {len(raw_mem)} embeddings")

    # 4) CPU threshold compute
    print("▶ Computing threshold on CPU (this can take a while)…")
    detector_cpu = KNNDetector(k=cfg["model"]["k"])
    detector_cpu.fit(raw_mem)
    mem_dists = [float(detector_cpu.score(x)) for x in tqdm(
        raw_mem,
        desc="  - thresh dists",
        unit="emb",
        file=sys.stdout,
        leave=False
    )]
    pct = cfg.get("threshold", {}).get("percentile", 90)
    threshold = float(np.percentile(mem_dists, pct))
    print(f"  ✔ Threshold @ {pct}-percentile = {threshold:.6f}")

    # 5) build GPU bank
    mem_bank = torch.stack([x.squeeze(0) for x in raw_mem], dim=0).to(device)
    print(f"▶ Moved memory bank to GPU: shape {tuple(mem_bank.shape)}")
    K = cfg["model"]["k"]

    # 6) glob eval_data/test WAVs
    root    = cfg["data"]["root"]
    pattern = f"{root}/eval_data/raw/*/test/**/*.wav"
    wavs    = sorted(glob.glob(pattern, recursive=True))
    print(f"▶ Glob pattern: {pattern}")
    print(f"▶ Found {len(wavs)} WAVs for evaluation")
    if not wavs:
        print("⚠️  No eval_data test files found! Check download and folder names.")
        sys.exit(1)

    # 7) inference with tqdm
    print("▶ Running inference with GPU KNN scoring…")
    writers = {}
    for path in tqdm(
        wavs,
        desc="  Scoring clips",
        unit="clip",
        file=sys.stdout,
        leave=True,
        dynamic_ncols=True,
    ):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device); sr = int(sr)

        feat = backbone(wav, sr)                  # (1,D) tensor
        d    = torch.cdist(feat, mem_bank)        # (1,N_train)
        topk = torch.topk(d, k=K, dim=1, largest=False).values
        score = float(topk.mean().item())
        decision = 1 if score > threshold else 0

        p = Path(path)
        machine = p.parent.parent.name
        section = p.parent.name.split("_")[1]
        tag     = f"{machine}_section_{section}"

        if tag not in writers:
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (asc.open("w"), dec.open("w"))
            print(f"  ↪ Opening files for {tag}")

        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # 8) close
    for asc_fp, dec_fp in writers.values():
        asc_fp.close(); dec_fp.close()

    print(f"✅ Finished! CSVs are in {csv_dir.resolve()}")


if __name__ == "__main__":
    main()
