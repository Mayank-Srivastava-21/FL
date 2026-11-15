# Geometry- and History-Aware FHHD Aggregation for Federated Learning under Non-IID Data and Byzantine Perturbations

**Team: FL Explorer** — Pragati Nayak, Mayank Srivastava, Gaurav Kumar Jha\
Indian Institute of Science Education and Research (IISER) Bhopal, India

Code for the experimental study of three server-side aggregation rules in the **FHHD** family: **FHHD-Plus**, **FHHD-R**, and **FHHD-R+**. The rules are compared with FedAvg, Krum, coordinate-wise Trimmed Mean, a spatial-temporal baseline, and a BOBA-inspired baseline on four image benchmarks under four Byzantine attacks.

> **Scope note.** The FHHD variants are empirical aggregation heuristics. They come with **no** formal Byzantine-resilience or convergence guarantee. "FHHD" is only the project-family name: the code does **not** compute a classical Helmholtz–Hodge decomposition of a vector field. The "BOBA-inspired" baseline is a simplified heuristic, not the original BOBA algorithm.

---

## Overview

Honest clients with non-IID data can submit updates that look unusual, while Byzantine clients submit deliberately corrupted ones. FHHD scores each client update from several complementary angles instead of relying on one distance or similarity measure, then down-weights or gates suspicious clients.

All variants share one pipeline:

1. **Norm clipping** of every client update to `2 x` the median update norm.
2. **Four per-client signals**
   - *Centrality*: distance from the coordinate-wise median centre.
   - *Direction*: cosine alignment with a peer-derived reference direction.
   - *Peer agreement*: median pairwise cosine similarity with other clients.
   - *Temporal consistency*: deviation from a per-client exponential moving average (EMA) of past updates.
3. **Agreement gate** that rejects at most `f` clients (the configured Byzantine budget).
4. **Score floor, weight normalisation, weight floor**, then a weighted sum.
5. **Server-step norm cap** on the aggregate.

| Variant | How it combines the evidence |
|---|---|
| **FHHD-Plus** | Multiplicative fusion of the four signals (strict: one low signal lowers the weight). |
| **FHHD-R** | Geometric-mean fusion (balanced, less sensitive to a single low signal). |
| **FHHD-R+** | FHHD-R aggregate blended with a coordinate-wise trimmed mean: `0.75 * FHHD-R + 0.25 * Trimmed`. |

---

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | Entry point. Runs the full sweep (datasets x attacks x methods), evaluates, and writes CSV results. |
| `config.py` | All hyperparameters, the dataset / attack / method lists, and output settings. |
| `aggregators.py` | FedAvg, Krum, Trimmed Mean, Spatial-Temporal, BOBA-inspired, FHHD-Plus, FHHD-R, FHHD-R+. |
| `attacks.py` | Byzantine attacks: `sign_flip`, `gaussian`, `model_replacement`, `ipm`. |
| `data_utils.py` | Dataset loading, 32x32 RGB preprocessing, and Dirichlet non-IID partitioning. |
| `models.py` | `SimpleCNN` used for all four datasets. |

---

## Installation

Python 3.8+ is recommended. A GPU is optional; the code falls back to CPU automatically.

```bash
git clone https://github.com/Mayank-Srivastava-21/FL.git
cd FL
pip install torch torchvision numpy pillow
```

---

## Datasets

By default `config.py` sets `data_root = "./data"` and `download = False`, so datasets must already be on disk (a missing dataset is skipped with a warning).

```
data/
├── cifar-10-batches-py/        # CIFAR-10 (torchvision layout)
├── cifar-100-python/           # CIFAR-100 (torchvision layout)
├── PACS/
│   └── <domain>/<class>/*.jpg  # art_painting, cartoon, photo, sketch
└── Office-Caltech10/
    └── <domain>/<class>/*.jpg  # amazon, caltech, dslr, webcam
```

- **CIFAR-10 / CIFAR-100:** set `download = True` in `config.py` to let torchvision fetch them, or place the standard archives under `./data`.
- **PACS / Office-Caltech10:** download manually and arrange as domain / class / image folders as shown above.

---

## Running the experiments

```bash
python main.py
```

This runs every dataset x attack x method combination (4 x 5 x 8 = **160 runs**). For a quick smoke test, set `max_runs = 5` in `config.py` first.

To change anything (datasets, attacks, seeds, Dirichlet alpha, Byzantine fraction, FHHD hyperparameters), edit `Config` in `config.py`.

### Outputs

Results are written to `results_full_sweep/` (configurable via `results_dir`):

| File | Contents |
|---|---|
| `sweep_summary.csv` | Final global accuracy, worst-class accuracy (WCA), macro accuracy per run. |
| `method_comparison.csv` | Cross-method comparison summary. |
| `round_metrics.csv` | Per-round accuracy, WCA, macro accuracy. |
| `class_accuracy.csv` | Per-class accuracy. |
| `aggregation_diagnostics.csv` | Per-round aggregation diagnostics. |
| `client_label_histograms.csv` | Class distribution of each client's partition. |
| `run_metadata.csv` | Seeds, Byzantine client IDs, partition sizes, and run settings. |

---

## Experimental setup

| Item | Value |
|---|---|
| Datasets | CIFAR-10, CIFAR-100, PACS, Office-Caltech10 |
| Input | RGB, resized to 32 x 32 |
| Model | `SimpleCNN` (Conv 3→16, ReLU, MaxPool, Conv 16→32, ReLU, MaxPool, FC 128, output layer) |
| Clients / Byzantine | N = 10 clients, f = 3 Byzantine (30%), IDs `{3, 6, 7}` |
| Partition | Dirichlet label split, alpha = 0.1 |
| Local training | SGD, lr 0.05, 2 local epochs, batch size 64 |
| Server | learning rate 1.0, 20 communication rounds |
| Seed | 3042 (partition and Byzantine IDs fixed across all methods) |
| Folder datasets | at most 150 images per class per domain, stratified 80/20 train/test split |

**Attacks**

| Attack | Byzantine update |
|---|---|
| `none` | No corruption. |
| `sign_flip` | The client's own local delta, negated. |
| `gaussian` | Honest-mean vector plus isotropic Gaussian noise (std = 0.5 x norm of the mean). |
| `model_replacement` | Random unit direction scaled to 5x the honest-mean norm (a high-norm proxy, **not** a backdoor attack). |
| `ipm` | Negative honest-mean vector, scale 1.0. |

**Metrics** (all in %): global accuracy (A), worst-class accuracy (WCA), and macro accuracy (M). A WCA of 0 means at least one represented test class received no correct predictions, not that the whole model failed.

**Key FHHD hyperparameters** (all in `config.py`): clip factor 2.0, centre temperature 1.5, direction temperature 5.0 / floor 0.05, agreement temperature 8.0, agreement gate 0.05, EMA momentum 0.80, temporal warm-up 2 / threshold 1.5 / strength 0.5, score floor 0.01, weight floor 0.02, server-step cap factor 2.0, FHHD-R+ blend 0.25.

---

## Results summary

Best method by **global accuracy** (%) in each setting. Full tables (including WCA and macro accuracy for all eight methods) are in the paper.

| Dataset | No attack | Sign flip | Gaussian | Model repl. | IPM |
|---|---|---|---|---|---|
| CIFAR-10 | BOBA-inspired (59.65) | **FHHD-R+** (42.81) | BOBA-inspired (48.23) | BOBA-inspired (48.32) | BOBA-inspired (49.45) |
| CIFAR-100 | **FHHD-R** (31.22) | **FHHD-Plus** (22.06) | **FHHD-Plus** (25.96) | **FHHD-Plus** (25.75) | **FHHD-R** (26.55) |
| PACS | Spatial-Temp. (32.93) | **FHHD-Plus** (19.44) | **FHHD-R** (26.25) | **FHHD-R** (24.67) | **FHHD-Plus** (26.37) |
| Office-Caltech10 | BOBA-inspired (40.12) | **FHHD-R** (16.60) | **FHHD-R** (29.84) | **FHHD-R** (29.05) | BOBA-inspired (26.09) |

**Takeaways**

- No single method is best across all datasets and attacks; the strongest method depends on the dataset and the perturbation.
- FHHD variants lead in 13 of the 20 dataset-attack settings, mostly under attack; baselines (BOBA-inspired, Spatial-Temporal) lead in the rest, including most no-attack settings.
- WCA is 0 in most attacked settings: a reasonable global accuracy can coexist with complete failure on at least one class.

---

## Limitations

- Empirical study only: no Byzantine-resilience or convergence guarantees.
- Single seed (3042), one Dirichlet alpha (0.1), one Byzantine fraction (30%), and a small CNN at 32 x 32 resolution.
- `model_replacement` is a random high-norm proxy, not a constructed malicious target model or backdoor.
- The BOBA-inspired comparator is a simplified heuristic, not the original BOBA algorithm.

---

## Citation

```bibtex
@misc{nayak2026fhhd,
  title  = {Geometry- and History-Aware FHHD Aggregation for Federated Learning under Non-IID Data and Byzantine Perturbations},
  author = {Nayak, Pragati and Srivastava, Mayank and Jha, Gaurav Kumar},
  year   = {2026},
  note   = {Team FL Explorer, Indian Institute of Science Education and Research Bhopal}
}
```

## Team contributions

| Member | Responsibilities |
|---|---|
| Pragati Nayak | Literature review, model architecture, no-attack and sign-flip evaluation, CIFAR-10/100 training and testing. |
| Mayank Srivastava | Literature review, model architecture, Gaussian-attack evaluation, PACS training and testing, source-code organisation. |
| Gaurav Kumar Jha | Literature review, model architecture, Office-Caltech10 training and testing, IPM-attack evaluation, post-processing. |

## References

1. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 (FedAvg).
2. Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*, NeurIPS 2017 (Krum).
3. Yin et al., *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*, ICML 2018.
4. Li et al., *Byzantine-Robust Federated Learning through Spatial-Temporal Analysis of Local Model Updates*, ICPADS 2021.
5. Bao et al., *BOBA: Byzantine-Robust Federated Learning with Label Skewness*, AISTATS 2024.
6. Krizhevsky and Hinton, *Learning Multiple Layers of Features from Tiny Images*, 2009.
7. Li et al., *Deeper, Broader and Artier Domain Generalization*, ICCV 2017 (PACS).
8. Gong et al., *Geodesic Flow Kernel for Unsupervised Domain Adaptation*, CVPR 2012 (Office-Caltech).
