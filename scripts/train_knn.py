#!/usr/bin/env python
"""
Extract BEATs embeddings for *normal* clips and build the k‑NN memory bank.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.models.detector import KNNDetector


def collate(batch):
    return batch  # variable‑length tensors; handle in loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DCASETask2Dataset(cfg["dcase_root"], split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate, num_workers=cfg["train"]["num_workers"])

    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    detector = KNNDetector(k=cfg["model"]["k"])

    feats = []
    paths = []
    for waveform, sr, path in tqdm(loader):
        waveform = waveform[0].to(device)
        feat = backbone(waveform, sr[0])
        feats.append(feat.cpu())
        paths.append(path[0])

    detector.fit(feats)

    # save memory bank
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"memory": feats, "paths": paths}, out_dir / "memory_bank.pt")
    print("Memory bank saved to", out_dir / "memory_bank.pt")


if __name__ == "__main__":
    main()
