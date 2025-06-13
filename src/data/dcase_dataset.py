"""
DCASE 2025 Task 2 dataset loader (dev + eval sets)
=================================================

Expected on-disk layout after running *download_task2_data.sh*::

    <dcase_root>/
      dev_data/raw/<machine>/<train|test|supplemental>/section_XX/*.wav
      eval_data/raw/<machine>/<train|test>/section_XX/*.wav

The loader is *recursive* and therefore agnostic to any extra folder
depth (``section_00``, ``id_00`` …).  It returns a **waveform tensor,
sample-rate, absolute-path** triple.  Labels are not used by the
unsupervised baseline, so we emit **-1** for *test* clips and **0** for
*train* clips simply to keep the signature compatible with supervision
if you need it later.

Example
-------
>>> ds = DCASETask2Dataset("data/dcase2025t2", split="train")
>>> wav, sr, path = ds[0]
>>> wav.shape, sr, path
(torch.Size([1, 160000]), 16000, '.../ToyCar/section_00/...wav')
"""

from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import torchaudio


class DCASETask2Dataset(Dataset):
    """
    Recursively collects WAVs matching
        <root>/<dev|eval>_data/raw/**/<train|test>/**.wav
    so it works for 'section_00', 'id_00/section_01', etc.
    """

    def __init__(self, root: str, split: str = "train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.files = []
        for stage in ("dev_data", "eval_data"):
            patt = Path(root, stage, "raw", "**", split, "**", "*.wav")
            self.files += glob.glob(str(patt), recursive=True)

        if not self.files:
            raise RuntimeError(
                f"No wavs found beneath {root} for split='{split}'. "
                "Check folder names (train/test) and casing."
            )

        self.files.sort()

    # -------------------------------------------------------- #
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:            # mix to mono
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sr, path
