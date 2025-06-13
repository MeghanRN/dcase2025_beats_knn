#!/usr/bin/env python
"""
Self-supervised fine-tune of BEATs-Large on normal clips.
After 5 epochs we save `finetuned_beats_large.pt`.
"""
import argparse, torch, torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dcase_dataset import DCASETask2Dataset
from src.utils.file_utils import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_highperf.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DCASETask2Dataset(cfg["data"]["root"], split="train")
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                    shuffle=True, num_workers=cfg["train"]["num_workers"])

    bundle = getattr(torchaudio.pipelines, cfg["model"]["embedding"])
    hubert = bundle.get_model().to(device).train()

    opt = torch.optim.AdamW(hubert.parameters(), lr=cfg["finetune"]["lr"])

    for epoch in range(cfg["finetune"]["epochs"]):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['finetune']['epochs']}")
        for wav, sr, _ in pbar:
            wav = wav.to(device)
            loss = hubert(
                wav,
                mask=True,
                mask_prob=cfg["finetune"]["mask_ratio"]
            ).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(hubert.state_dict(), "finetuned_beats_large.pt")
    print("✓ saved finetuned_beats_large.pt")

if __name__ == "__main__":
    main()
