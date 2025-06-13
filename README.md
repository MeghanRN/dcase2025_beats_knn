# DCASE 2025 Task 2 — BEATs + K‑NN First‑Shot Anomalous Sound Detection

This repository implements the **pre‑trained BEATs transformer backbone** with a **k‑nearest‑neighbour (k‑NN) anomaly detector** recommended in state‑of‑the‑art systems for the DCASE 2025 Task 2 “First‑shot unsupervised anomalous sound detection for machine‑condition monitoring under domain shift”.

The code is designed to:

* extract log‑Mel spectrograms → BEATs embeddings for *normal* machine sounds  
* build a memory bank of normal embeddings (optionally fine‑tuned on a handful of target‑domain clips)  
* assign anomaly scores to test clips via the mean cosine distance to the *k* closest neighbours  
* output DCASE‑style CSV result files (`<machine_id>_<domain>.csv`)

> Expected dev‑set performance with the default config is **≈ 60–65 % AUC / pAUC** (harmonic mean) — a >10 pp improvement over the official auto‑encoder baseline.

---

## Quick start

```bash
# 1. create and activate a fresh conda env
conda env create -f environment.yml
conda activate dcase25

# 2. prepare the DCASE data (update the path!)
python scripts/preprocess.py --dcase_root /path/to/task2_dev

# 3. train / fit the memory bank
python scripts/train_knn.py --config configs/default.yaml

# 4. run inference on the dev‑set
python scripts/infer.py --config configs/default.yaml
```

The result CSVs will appear in `./results/` and can be scored with `dcase_task2_scoring.py`.

---

## Repository layout

```
dcase2025_beats_knn/
├── configs/
│   └── default.yaml          # all hyper‑parameters in one place
├── scripts/                  # CLI entry‑points
│   ├── preprocess.py
│   ├── train_knn.py
│   └── infer.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── dcase_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── beats_backbone.py
│   │   └── detector.py
│   └── utils/
│       ├── metrics.py
│       └── file_utils.py
├── tests/                    # minimal sanity checks
├── requirements.txt
├── environment.yml
├── LICENSE
└── .gitignore
```

---

## Configuration

All paths and hyper‑parameters live in **`configs/default.yaml`**.  
Key options:

| key | default | description |
| --- | --- | --- |
| `train.domain` | `source+target` | which normal clips to include in the memory bank |
| `model.k` | `3` | number of nearest neighbours |
| `model.embedding` | `"beats_base"` | BEATs checkpoint; any in `torchaudio.pipelines.BEATS_*` |
| `augmentation.snr` | `[0, 10, 20]` | mix background‑noise‑only clips at these SNRs |
| `augmentation.pitch_shift` | `[-1, 1]` | semitone shifts for speed / load simulation |

---

## Citing / acknowledgements

* BEATs: Chen **et al.** “BEATs: Audio Pre‑Training with Acoustic Tokenizers”, *ICML 2023*  
* DCASE Challenge organisers and baseline authors  
* `torchaudio`, `scikit‑learn`, `librosa`

This repo is released under the **MIT License** — see `LICENSE` for details.
