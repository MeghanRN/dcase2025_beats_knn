#!/usr/bin/env python
"""
Final inference script for DCASE-2025 Task-2 submission (eval set only).

Creates BOTH:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
for every machine / section in the *evaluation* test set.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch, torchaudio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.file_utils     import load_config
from src.data.dcase_dataset   import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.models.detector      import KNNDetector


def percentile_threshold(distances: list[float], pct: float) -> float:
    """Return the pct-ile of the list (e.g. 90 → 90th percentile)."""
    return float(np.percentile(np.array(distances), pct))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    # load config & device
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # locate bank and output dirs
    bank_dir = Path(cfg.get("output_dir",  cfg["logging"]["bank_out"]))
    csv_dir  = Path(cfg.get("csv_out_dir", cfg["logging"]["csv_out_dir"]))
    ensure_dir(csv_dir)

    # load backbone & memory bank
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    ckpt     = torch.load(bank_dir/"memory_bank.pt", map_location=device)

    # fit detector on the saved memory
    detector = KNNDetector(k=cfg["model"]["k"])
    detector.fit(ckpt["memory"])  # list of Tensors

    # derive decision threshold from training-bank distances
    mem_dists = [float(detector.score(f)) for f in ckpt["memory"]]
    thr_pct   = cfg.get("threshold", {}).get("percentile", 90)
    threshold = percentile_threshold(mem_dists, thr_pct)
    print(f"Decision threshold @ {thr_pct}-percentile: {threshold:.4f}")

    # prepare only the *evaluation* test set
    test_ds = DCASETask2Dataset(cfg["data"]["root"], split="test")
    # filter out dev_data/test, keep only eval_data/test
    test_ds.files = [
        f for f in test_ds.files
        if "/eval_data/" in f.replace("\\", "/")
    ]
    print(f"▶ Scoring {len(test_ds)} eval-set clips → CSV in `{csv_dir}`")

    loader = DataLoader(
        test_ds,
        batch_size   = 1,
        shuffle      = False,
        collate_fn   = lambda b: b,  # return list of 1 tuple
        num_workers  = cfg["dataloader"]["num_workers"],
        pin_memory   = device.type == "cuda",
    )

    # file handles per machine_section
    writers: dict[str, tuple] = {}

    # run inference with live tqdm bar
    for batch in tqdm(
        loader,
        desc="Scoring evaluation clips",
        unit="clip",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    ):
        wav, sr, path = batch[0]
        wav = wav.to(device)
        sr  = int(sr) if torch.is_tensor(sr) else sr

        # compute score & decision
        feat     = backbone(wav, sr)
        score    = float(detector.score(feat.cpu()))
        decision = 1 if score > threshold else 0

        p       = Path(path)
        machine = p.parts[-3]                    # e.g. "CoffeeGrinder"
        section = p.stem.split("_")[1]           # e.g. "00"
        tag     = f"{machine}_section_{section}"

        # open CSVs on first encounter
        if tag not in writers:
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (asc.open("w"), dec.open("w"))

        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # close all file handles
    for asc_fp, dec_fp in writers.values():
        asc_fp.close()
        dec_fp.close()

    print(f"✅  All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
