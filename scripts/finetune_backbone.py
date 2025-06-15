#!/usr/bin/env python
"""
Self‐supervised feature‐distillation fine‐tuning of HuBERT/BEATs.
No mask‐param required—works with any torchaudio pipeline.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset


def collate(batch):
    # batch is a list of one tuple: [(wav, sr, path)]
    return batch


def main():
    # 1. args & config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    # 2. paths & device
    data_root = cfg["data"]["root"]
    print(f"▶ Fine‐tuning on data: {data_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    # 3. hyperparams (hard‐coded here)
    EPOCHS      = 5
    LR          = 1e-5
    BATCH_SIZE  = 1
    NUM_WORKERS = cfg["train"]["num_workers"]

    # 4. dataset & loader
    ds = DCASETask2Dataset(data_root, split="train")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=(device.type=="cuda"),
    )

    # 5. teacher & student models
    bundle   = getattr(torchaudio.pipelines, cfg["model"]["embedding"])
    teacher  = bundle.get_model().to(device).eval()
    student  = bundle.get_model().to(device).train()

    # 6. optimizer
    optim = torch.optim.AdamW(student.parameters(), lr=LR)

    # 7. fine‐tune loop
    for epoch in range(1, EPOCHS+1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            wav, sr, _ = batch[0]
            wav = wav.to(device)

            # 7a. extract teacher features
            out_t = teacher.extract_features(wav)
            feats_t = out_t[0][-1].mean(dim=1)    # (1, D)

            # 7b. extract student features
            out_s = student.extract_features(wav)
            feats_s = out_s[0][-1].mean(dim=1)    # (1, D)

            # 7c. MSE loss
            loss = F.mse_loss(feats_s, feats_t)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 8. save student weights
    torch.save(student.state_dict(), "finetuned_beats_large.pt")
    print("✅ Saved finetuned_beats_large.pt")


if __name__ == "__main__":
    main()
