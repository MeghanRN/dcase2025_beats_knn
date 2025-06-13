"""
Minimal PyTorch Dataset for DCASE Task 2 dev‑set.

Assumes the directory hierarchy:
  <dcase_root>/<machine_type>/<machine_id>/<domain>/normal/<wav files>

and equivalent for `test`.
"""
import glob
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchaudio


class DCASETask2Dataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.meta = []

        root = Path(root)
        for wav in root.rglob("*/normal/*.wav"):
            if split == "train" and "test" in wav.parts:
                continue
            if split == "test" and "test" not in wav.parts:
                continue
            self.meta.append(wav)

        self.meta.sort()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        wav_path = self.meta[idx]
        waveform, sr = torchaudio.load(wav_path)
        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # target label: 0 = normal (train), unknown (-1) for test
        label = 0 if "normal" in wav_path.parts else -1
        return waveform, sr, str(wav_path)
