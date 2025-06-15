#!/usr/bin/env python
"""
Self‐supervised masked‐prediction fine‐tuning of the HuBERT/BEATs backbone.
Uses only configs/default.yaml to locate the data.
Saves finetuned_beats_large.pt in the repo root when done.
"""

import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml",
                    help="Path to your existing default.yaml")
    args = ap.parse_args()

    # 1) LOAD CONFIG & DATA ROOT
    cfg = load_config(args.config)
    data_root = cfg["data"]["root"]
    print(f"▶ Loading data from: {data_root}")

    # 2) Hyper‐parameters (defaults here; no need to edit default.yaml)
    FT_EPOCHS     = 5           # number of fine‐tune epochs
    MASK_RATIO    = 0.3         # masked‐prediction probability
    LR            = 1e-5        # learning rate for fine‐tuning
    BATCH_SIZE    = cfg["train"]["batch_size"]
    NUM_WORKERS   = cfg["train"]["num_workers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    # 3) DATASET & DATALOADER
    ds = DCASETask2Dataset(data_root, split="train")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 4) MODEL INIT
    #   Use the same backbone you have configured
    bundle = getattr(torchaudio.pipelines, cfg["model"]["embedding"])
    model = bundle.get_model().to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 5) FINE‐TUNE LOOP
    for epoch in range(1, FT_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{FT_EPOCHS}")
        for wav, sr, _ in pbar:
            wav = wav.to(device)

            # masked‐prediction forward returns an object with .loss
            out = model(wav, mask=True, mask_prob=MASK_RATIO)
            loss = getattr(out, "loss", out[0] if isinstance(out, (list, tuple)) else out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 6) SAVE WEIGHTS
    torch.save(model.state_dict(), "finetuned_beats_large.pt")
    print("✅ Saved finetuned_beats_large.pt")

if __name__ == "__main__":
    main()
