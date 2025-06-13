import torch
from src.models.beats_backbone import BEATsBackbone
from src.models.detector import KNNDetector
import numpy as np


def test_detector():
    torch.manual_seed(0)
    dummy = [torch.randn(10, 768) for _ in range(5)]
    det = KNNDetector(k=3)
    det.fit(dummy)
    score = det.score(torch.randn(8, 768))
    assert isinstance(score, float)


if __name__ == "__main__":
    test_detector()
    print("✓ all tests passed")
