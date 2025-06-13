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
    """Recursive loader for Task-2 *train* or *test* subsets."""

    def __init__(self, root: str, split: str = "train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.files: List[Path] = []

        root = Path(root)
        stages = ["dev_data", "eval_data"]
        domains = {"train": "train", "test": "test"}

        for stage in stages:
            base = root / stage / "raw"
            pattern = base / "**" / domains[split] / "**" / "*.wav"
            # recursive=True lets us match section_00/<file>.wav, etc.
            self.files += list(pattern.glob(recursive=True))

        if not self.files:
            raise RuntimeError(f"No wavs found in {root} for split='{split}'")

        self.files.sort()  # deterministic order

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        wav_path = self.files[idx]
        waveform, sr = torchaudio.load(wav_path)

        # convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # label is unused by k-NN baseline; keep for future extensions
        label = 0 if "/train/" in wav_path.as_posix() else -1
        return waveform, sr, str(wav_path)
