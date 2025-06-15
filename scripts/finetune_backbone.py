#!/usr/bin/env python
"""
Self‐supervised masked‐prediction fine‐tuning of the HuBERT/BEATs backbone.
Processes one clip at a time to avoid variable‐length stacking issues.
Writes finetuned_beats_large.pt in the repo root when done.
"""

import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset


def collate(batch):
    """
    Identity collate: returns a list of tuples [(wav, sr, path), ...].
    We use batch_size=1, so each batch is a list of length 1.
    """
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Your existing default.yaml with data.root and model settings"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    # 1) Where the data lives
    data_root = cfg["data"]["root"]
    print(f"▶ Fine-tuning on normal clips from: {data_root}")

    # 2) Hyper‐parameters (hard‐coded here; you can tweak if you like)
    FT_EPOCHS   = 5
    MASK_RATIO  = 0.3
    LR          = 1e-5
    BATCH_SIZE  = 1             # one clip at a time
    NUM_WORKERS = cfg["train"]["num_workers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    # 3) Dataset & DataLoader (batch_size=1 + identity collate)
    ds = DCASETask2Dataset(data_root, split="train")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    # 4) Load the same backbone used downstream
    bundle = getattr(torchaudio.pipelines, cfg["model"]["embedding"])
    model = bundle.get_model().to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 5) Fine‐tune loop
    for epoch in range(1, FT_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{FT_EPOCHS}")
        for batch in pbar:
            wav, sr, _ = batch[0]      # unpack the single-item batch
            wav = wav.to(device)

            out = model(wav, mask=True, mask_prob=MASK_RATIO)
            # some models return a .loss, others a list/tuple
            loss = getattr(out, "loss", out[0] if isinstance(out, (list, tuple)) else out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 6) Save the fine‐tuned weights
    torch.save(model.state_dict(), "finetuned_beats_large.pt")
    print("✅ Saved finetuned_beats_large.pt")


if __name__ == "__main__":
    main()
