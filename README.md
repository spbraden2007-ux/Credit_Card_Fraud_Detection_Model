# Credit Card Fraud Detection

> Hybrid Python pipeline for credit-card fraud detection combining statistical anomaly detection and supervised gradient boosting, tuned with Bayesian optimization (Optuna)

A semi-supervised approach: mine pseudo-labels via **EllipticEnvelope**, train **LightGBM** with **5-fold CV**, then **OR-ensemble** the two. Reproducible, configurable, and export-ready. **F1 Score ≈ 93%**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-3.x-green)
![Optuna](https://img.shields.io/badge/Optuna-TPE-purple)

---

## 🎯 What this repo delivers
- **Hybrid model**: EllipticEnvelope (unsupervised) + LightGBM (supervised)
- **Semi-supervised training**: top-k anomaly mining → pseudo-labels
- **Optuna tuning**: TPE sampler, 500 trials (configurable)
- **5-Fold Stratified CV** with macro-F1
- **Single command** training & **CSV export** for submission
- **Config-driven** paths/hyperparams for full reproducibility

---

## 🧭 Architecture

```
Raw CSVs ──► EllipticEnvelope (score_samples) ──► top-k pseudo labels
│ │
└───────────── OR ensemble ◄──────────┘
▲
LightGBM (Optuna-tuned, CV)
│
submission.csv (ID, Class)
```
---

## 📦 Repository Structure
```
credit-card-fraud-detection/
├── src/
│ ├── train.py # main training/optimization/ensemble
│ ├── infer.py # (optional) pure inference on test
│ ├── models/
│ │ ├── envelope.py # EllipticEnvelope wrapper (top-k)
│ │ └── lgbm.py # LightGBM + CV + Optuna hooks
│ └── utils.py # seed, metrics, io helpers
├── data/
│ ├── raw/ # train.csv, val.csv, test.csv
│ └── processed/ # (optional) cached artifacts
├── experiments/ # study.json, cv_scores.csv, submission.csv
├── notebooks/EDA_and_benchmark.ipynb
├── config.yaml
├── requirements.txt
├── LICENSE
└── README.md
```
---

## 🚀 Quick Start

### 1) Install
```bash
git clone https://github.com/spbraden2007-ux/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### 2) Place data
```
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```
### 3) Configure

```
# config.yaml
data:
  train_path: data/raw/train.csv
  val_path:   data/raw/val.csv
  test_path:  data/raw/test.csv
training:
  seed: 42
  n_splits: 5
  optuna_trials: 500
  top_k_train: 118
  top_k_test: 314
model:
  envelope:
    support_fraction: 0.994
  lgbm:
    boosting_type: dart
    learning_rate: 0.3
    n_estimators: 270
```
### 4) Run

```
python src/train.py --config config.yaml --output-dir experiments/run1
# ➜ experiments/run1/submission.csv 생성
```

