#!/usr/bin/env python
"""
Final inference script for DCASE-2025 Task-2 submission.

Creates BOTH:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
for every machine / section in the evaluation test set.
"""

from __future__ import annotations
from pathlib import Path
import argparse, math, yaml, glob

import numpy as np
import torch, torchaudio
from tqdm import tqdm

from src.utils.file_utils    import load_config
from src.data.dcase_dataset  import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.models.detector     import KNNDetector


# ─────────────────────────── helpers ──────────────────────────
def percentile_threshold(distances: list[float], pct: float) -> float:
    """Return pct-ile of the list (e.g. 90 → 0.9)."""
    return float(np.percentile(np.array(distances), pct))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── main ────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # legacy vs new keys
    bank_dir = Path(cfg.get("output_dir",  cfg["logging"]["bank_out"]))
    csv_dir  = Path(cfg.get("csv_out_dir", cfg["logging"]["csv_out_dir"]))
    ensure_dir(csv_dir)

    # ── load backbone & detector ───────────────────────────────
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()

    ckpt = torch.load(bank_dir / "memory_bank.pt", map_location=device)
    detector = KNNDetector(k=cfg["model"]["k"])
    detector.fit(ckpt["memory"])                 # list[Tensor]

    # derive threshold
    mem_dists = [detector.score(f).item() for f in ckpt["memory"]]
    thr_pct   = cfg.get("threshold", {}).get("percentile", 90)
    threshold = percentile_threshold(mem_dists, thr_pct)
    print(f"Decision threshold @ {thr_pct}-percentile: {threshold:.4f}")

    # ── dataset (evaluation test set) ──────────────────────────
    test_ds = DCASETask2Dataset(cfg["data"]["root"], split="test")
    loader  = torch.utils.data.DataLoader(
        test_ds,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = lambda b: b,        # keep variable lengths
        num_workers = cfg["dataloader"]["num_workers"],
        pin_memory  = device.type == "cuda",
    )

    # writers keyed by <machine>_<section>
    writers = {}

    for batch in tqdm(loader, desc="Scoring evaluation clips"):
        wav, sr, path = batch[0]
        wav = wav.to(device)
        sr  = int(sr) if torch.is_tensor(sr) else sr

        score = detector.score(backbone(wav, sr))
        score = float(score if isinstance(score, float) else score.item())
        decision = 1 if score > threshold else 0

        p = Path(path)
        # example path: .../eval_data/raw/ToyRCCar/test/section_00_*.wav
        machine = p.parts[-3]                  # ToyRCCar
        section = p.stem.split("_")[1]         # 00

        tag = f"{machine}_section_{section}"
        if tag not in writers:
            # open two CSV files (append mode makes reruns safe)
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (
                asc.open("w"),
                dec.open("w"),
            )
        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # close all
    for asc_fp, dec_fp in writers.values():
        asc_fp.close()
        dec_fp.close()

    print(f"✅  All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
