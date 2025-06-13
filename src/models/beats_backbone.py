from pathlib import Path
import torch, torchaudio


class BEATsBackbone(torch.nn.Module):
    """
    • Works with waveform-based bundles (HuBERT, Wav2Vec2, XLSR, …)
    • Works with spectrogram-based bundles (BEATs) if you switch embedding.
    """

    def __init__(self, checkpoint: str = "HUBERT_BASE"):
        super().__init__()

        self.bundle = getattr(torchaudio.pipelines, checkpoint)
        self.model = self.bundle.get_model().eval()
        self.sample_rate = self.bundle.sample_rate

        # Decide whether the bundle expects waveform or log-mel input
        self.expect_waveform = checkpoint.startswith(
            ("HUBERT", "WAV2VEC", "XLSR")
        )

        if not self.expect_waveform:
            # Mel layer only needed for BEATs-style models
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
            )

    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, wav: torch.Tensor, sr: int | torch.Tensor) -> torch.Tensor:
        """
        wav : (1, T) float32, −1…1
        sr  : sample-rate (int or 0-D tensor)
        returns (1, D) clip-level embedding
        """
        sr = int(sr) if torch.is_tensor(sr) else sr
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if cfg["model"].get("use_layer_stack", False) and isinstance(x, list):
            feats = torch.cat(x[::3], dim=-1)    # layers 0,3,6,9  (B,T,4*1024)

        # ── extract features ─────────────────────────────────────
        if self.expect_waveform:
            out = self.model.extract_features(wav)   # tuple or list
        else:
            mel = self.mel(wav)                      # (1, F, T)
            out = self.model.extract_features(mel)

        # Normalise heterogeneous return types -> Tensor (B, T, D)
        if isinstance(out, tuple):       # Wav2Vec2 returns (list, lengths)
            x = out[0]
        else:                            # HuBERT returns list[Tensor]
            x = out

        if isinstance(x, list):
            feats = x[-1]                # last layer = highest rep.
        else:
            feats = x

        return feats.mean(dim=1)         # time-avg pooling  (B, D)
