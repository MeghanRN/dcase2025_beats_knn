"""
BEATs backbone wrapper.

Loads a pre‑trained BEATs model from torchaudio and exposes a
`forward(waveform, sr)` method that returns an embedding tensor
of shape (num_frames, embed_dim).

If `fine_tune=True`, all transformer parameters are trainable.
"""
from typing import Tuple

import torch
import torchaudio


class BEATsBackbone(torch.nn.Module):
    def __init__(self, checkpoint: str = "BEATS_BASE", fine_tune: bool = False):
        super().__init__()
        try:
            self.bundle = getattr(torchaudio.pipelines, checkpoint)
        except AttributeError as e:
            raise ValueError(f"Unknown BEATs checkpoint: {checkpoint}") from e

        self.model = self.bundle.get_model()
        if not fine_tune:
            for p in self.model.parameters():
                p.requires_grad = False

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.bundle.sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=128,
            f_min=0,
            f_max=None
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Compute frame‑level embeddings.

        Args:
            waveform: (1, num_samples) mono tensor in range [-1,1]
            sr: sampling rate

        Returns:
            emb: (T, D) tensor where T ≈ num_frames
        """
        if sr != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.bundle.sample_rate
            )

        # (128, num_frames)
        mel = self.mel_spec(waveform).log1p()
        # BEATs expects (batch=1, frames, mel_bins)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)
        emb = self.model.extract_features(mel)[0]  # (1, T, D)
        return emb.squeeze(0)
