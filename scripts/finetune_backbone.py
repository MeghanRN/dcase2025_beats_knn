#!/usr/bin/env python
"""
Self-supervised fine-tune of BEATs-Large on normal clips.
Saves finetuned_beats_large.pt for later use by train_knn.py.
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
    ap.add_argument("--config", default="configs/default_highperf.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Use the literal path from dcase_root
    data_root = cfg.get("dcase_root") or cfg["data"]["root"]
    print(f"▶ Fine-tuning on data in: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train set: only 'train' split normals
    ds = DCASETask2Dataset(data_root, split="train")
    dl = DataLoader(
        ds,
        batch_size=cfg["finetune"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )

    # Load BEATs-Large (or whichever) bundle
    bundle = getattr(torchaudio.pipelines, cfg["model"]["embedding"])
    model = bundle.get_model().to(device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["finetune"]["lr"])

    for epoch in range(cfg["finetune"]["epochs"]):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['finetune']['epochs']}")
        for wav, sr, _ in pbar:
            wav = wav.to(device)
            # masked-prediction forward returns an object with .loss
            out = model(wav, mask=True, mask_prob=cfg["finetune"]["mask_ratio"])
            loss = out.loss if hasattr(out, "loss") else out[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), "finetuned_beats_large.pt")
    print("✓ Saved finetuned_beats_large.pt")


if __name__ == "__main__":
    main()
