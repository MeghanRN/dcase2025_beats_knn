"""
Backbone wrapper: works with HuBERT, Wav2Vec2, BEATs.
"""

from pathlib import Path
import torch, torchaudio


class BEATsBackbone(torch.nn.Module):
    def __init__(self, checkpoint="HUBERT_BASE", use_layer_stack=False):
        super().__init__()

        self.bundle = getattr(torchaudio.pipelines, checkpoint)
        self.model = self.bundle.get_model().eval()
        self.sample_rate = self.bundle.sample_rate
        self.use_layer_stack = use_layer_stack

        # waveform vs. spectrogram input
        self.expect_waveform = checkpoint.startswith(
            ("HUBERT", "WAV2VEC", "XLSR")
        )
        if not self.expect_waveform:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
            )

        # load optional fine-tuned weights
        finetuned = Path("finetuned_beats_large.pt")
        if finetuned.exists():
            self.model.load_state_dict(
                torch.load(finetuned, map_location="cpu"), strict=False
            )
            print("✔ loaded fine-tuned backbone")

    # -------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, wav: torch.Tensor, sr: int | torch.Tensor):
        sr = int(sr) if torch.is_tensor(sr) else sr
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # extract features
        if self.expect_waveform:
            out = self.model.extract_features(wav)
        else:
            mel = self.mel(wav)
            out = self.model.extract_features(mel)

        # normalise return types
        x = out[0] if isinstance(out, tuple) else out
        if self.use_layer_stack and isinstance(x, list):
            feats = torch.cat(x[::3], dim=-1)      # concat layers 0,3,6,9
        else:
            feats = x[-1] if isinstance(x, list) else x

        return feats.mean(dim=1)                   # (B, D)
