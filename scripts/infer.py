#!/usr/bin/env python
"""
Run inference on test clips and write DCASE CSV files.
"""
import argparse
from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.models.detector import KNNDetector


def collate(batch):
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DCASETask2Dataset(cfg["dcase_root"], split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate, num_workers=2)

    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()

    # load detector
    ckpt = torch.load(Path(cfg["output_dir"]) / "memory_bank.pt", map_location=device)
    detector = KNNDetector(k=cfg["model"]["k"])
    detector.fit(ckpt["memory"])

    results_dir = Path(cfg["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    paths = []
    for waveform, sr, path in tqdm(loader):
        waveform = waveform[0].to(device)
        feat = backbone(waveform, sr[0])
        score = detector.score(feat)
        scores.append(score)
        paths.append(path[0])

    # group by <machine_id_domain>
    rows = {}
    for p, s in zip(paths, scores):
        parts = Path(p).parts
        machine_id = parts[-4]  # <machine_type>/<machine_id>/<domain>/...
        domain = parts[-3]
        key = f"{machine_id}_{domain}"
        rows.setdefault(key, []).append((p, s))

    for key, items in rows.items():
        csv_path = results_dir / f"{key}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "anomaly_score"])
            for p, s in items:
                writer.writerow([Path(p).name, s])
        print("Wrote", csv_path)


if __name__ == "__main__":
    main()
