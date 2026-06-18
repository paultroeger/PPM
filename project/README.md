# Private ML: PATE vs. DP-SGD vs. SGD

Comparison of three training methods on MNIST, SVHN, and CIFAR-10 under matched
privacy budgets (ε ∈ {1, 4, 8}, δ = 1e-5):

- **PATE** — private knowledge transfer via a noisy teacher-ensemble vote (GNMax).
- **DP-SGD** — differentially private SGD via [Opacus](https://opacus.ai/).
- **SGD** — non-private baseline.

## Demo

`demo.html` is a standalone, exported notebook (`pate_explorer.ipynb` -
*"How much accuracy does privacy cost?"*) that visualizes the privacy–utility
trade-offs from these experiments.

## Setup

```bash
pip install torch torchvision opacus numpy scipy sympy absl-py
```

Datasets download automatically to `/data` on first run.

## DP-SGD / SGD

Single batch runner over the full grid (datasets × epochs × ε):

```bash
python run_sgd_dpsgd.py --only dpsgd --model cnn # DP-SGD, per-dataset CNN
python run_sgd_dpsgd.py --only sgd # non-private baseline
python run_sgd_dpsgd.py --only dpsgd --model both # CNN then MLP
```

- `--only` : `sgd` | `dpsgd` | `both`
- `--model`: `cnn` | `simple` (MLP) | `both`

Run configurations are configured at the top of `run_sgd_dpsgd.py`. Results are written
as per-run logs, a `*_metrics.json`, and a `summary.tsv`.

## PATE

Teacher checkpoints are **not** included in the repo, they are trained on first
run and written to `--train_dir` (default `/tmp/pate_pytorch_train`).

**Single end-to-end run** (`pate.py`), trains the teacher ensemble, then the
student, then reports the epsilon:

```bash
python pate.py --dataset mnist --nb_teachers 100 \
    --teacher_epochs 100 --student_epochs 100 --student_queries 1000 \
    --noise_scale 31 --delta 1e-5
```

**Automated ε sweep** (`pate_auto_sweep.py`), trains/resumes teacher ensembles,
auto-selects noise scales for target ε ∈ {1,4,8}, trains one student per σ, and
writes per-run logs and a `summary.tsv` under `logs/`:

```bash
python pate_auto_sweep.py --datasets mnist,svhn,cifar10 \
    --teachers 50,100,200 --epochs 50,100,150
```

## Notes

- `aggregation.py`, `core.py`, `metrics.py`, `smooth_sensitivity.py` are almost 1:1 to the original tensorflow implementation (see: [PATE 2017](https://github.com/tensorflow/privacy/tree/master/research/pate_2017) and [PATE 2018](https://github.com/tensorflow/privacy/tree/master/research/pate_2018)).
- `deep_cnn.py`, `gnmax_accountant.py`, `pate.py`, `test_load_data.py`, `train_student.py`, `train_teachers.py` are our PyTorch port of the PATE implementation.
- The PATE student is trained purely supervised (no semi-supervised / GAN student), to keep a clean comparison with DP-SGD.
- `compute_normalize.py` computes the per-channel normalization statistics (mean/std, from each dataset's train split) that are hard-coded into the loaders in `test_load_data.py`.
- `sgd_vs_dpsgd.py` is a small standalone training run of SGD vs. DP-SGD.
