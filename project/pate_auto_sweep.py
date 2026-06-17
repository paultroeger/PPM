"""
Automated PATE noise-scale sweep.

Trains or resumes teacher ensembles, chooses noise scales for target epsilons,
trains one student per chosen sigma, and writes logs plus markdown summaries.

Note: Claude was used to help create this multi-training script.
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime

import numpy as np
import torch

import aggregation
import core
import deep_cnn
import gnmax_accountant
import metrics
import test_load_data
from train_student import ensemble_preds

DATASETS       = ["mnist", "svhn", "cifar10"]
NB_TEACHERS    = [50, 100, 200]
TEACHER_EPOCHS = [50, 100, 150]
STUDENT_EPOCHS = 100
STUDENT_QUERIES = 1000
DELTA          = 1e-5
BATCH_SIZE     = 128
LR             = 0.01
TARGET_EPS     = [1.0, 4.0, 8.0]
EPS_TOL        = 0.1

LD = {"mnist": test_load_data.ld_mnist, "svhn": test_load_data.ld_svhn,
      "cifar10": test_load_data.ld_cifar10}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
TEACHER_BASE = os.path.join(DATA_DIR, "pate_teachers")


def md_path(te):
    """Markdown path for one teacher-epoch setting."""
    q = "" if STUDENT_QUERIES == 1000 else f"_q{STUDENT_QUERIES}"
    return os.path.join(DATA_DIR, f"pate_best_noise_e{te}{q}.md")

SIGMA_GRID = sorted(set(
    list(range(5, 30, 1)) + list(range(30, 80, 2)) +
    list(range(80, 160, 5)) + list(range(160, 320, 10))
))

ORDERS = gnmax_accountant.default_orders()
_DI_CACHE = {}


def log(msg=""):
    print(msg, flush=True)


def train_teachers(dataset, nb_teachers, train_dir, epochs):
    """Trains (or resumes) the ensemble. Loads the dataset once, partitions it."""
    os.makedirs(train_dir, exist_ok=True)
    train_data, train_labels, _, _ = LD[dataset](data_dir=DATA_DIR)

    for tid in range(nb_teachers):
        ckpt = os.path.join(train_dir, f"{dataset}_{nb_teachers}_teachers_{tid}.pt")
        if os.path.exists(ckpt):
            continue
        data, labels = test_load_data.partition_dataset(
            train_data, train_labels, nb_teachers, tid)
        log(f"    teacher {tid}/{nb_teachers - 1} (n={len(labels)})")
        deep_cnn.train(dataset, data, labels, ckpt,
                       epochs=epochs, batch_size=BATCH_SIZE, lr=LR)


def sweep_epsilon(clean_votes):
    """Cheap no-smooth-sensitivity epsilon curve used to seed bisection."""
    pairs = []
    for sigma in SIGMA_GRID:
        try:
            eps = gnmax_accountant.compute_epsilon(
                clean_votes, sigma=float(sigma), delta=DELTA,
                include_smooth_sensitivity=False)["epsilon"]
        except (ValueError, FloatingPointError):
            continue  # the accountant can choke on extreme sigma
        if np.isfinite(eps):
            pairs.append((float(sigma), float(eps)))
    return pairs


def di_epsilon(sigma, num_queries):
    """Data-independent epsilon for GNMax queries at one sigma."""
    rdp = num_queries * core.rdp_data_independent_gaussian(float(sigma), ORDERS)
    eps, _ = core.compute_eps_from_delta(ORDERS, rdp, DELTA)
    return float(eps)


def fully_data_independent(sigma, num_teachers, num_classes):
    """Whether the votes-free epsilon is exact for this setting."""
    return bool(np.all(core.is_data_independent_always_opt_gaussian(
        num_teachers, num_classes, float(sigma), ORDERS)))


def di_solve(target, num_queries):
    """Bisect the cached data-independent epsilon curve."""
    key = (target, num_queries)
    if key in _DI_CACHE:
        return _DI_CACHE[key]
    lo, hi = 1.0, 1000.0
    while di_epsilon(hi, num_queries) > target and hi < 1e6:
        hi *= 2
    while di_epsilon(lo, num_queries) < target and lo > 1e-3:
        lo *= 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if di_epsilon(mid, num_queries) > target:
            lo = mid
        else:
            hi = mid
    sigma = round(0.5 * (lo + hi), 2)
    res = (sigma, di_epsilon(sigma, num_queries))
    _DI_CACHE[key] = res
    return res


def solve_sigma(clean_votes, pairs, target):
    """Pick sigma for target epsilon, using the fast path when exact."""
    num_queries, num_classes = clean_votes.shape
    num_teachers = int(np.sum(clean_votes[0]))
    sigma, di_eps = di_solve(target, int(num_queries))
    if fully_data_independent(sigma, num_teachers, int(num_classes)):
        return sigma, di_eps, True
    sigma, achieved = find_sigma(clean_votes, pairs, target)
    return sigma, achieved, False


def find_sigma(clean_votes, pairs, target, tol=EPS_TOL, max_iter=25):
    """Return a sigma whose exact epsilon is close to target."""
    def eps_ss(sigma):
        try:
            return gnmax_accountant.compute_epsilon(
                clean_votes, sigma=float(sigma), delta=DELTA,
                include_smooth_sensitivity=True)["epsilon"]
        except (ValueError, OverflowError):
            return float("inf")

    grid = sorted(pairs)
    # Use the final decreasing tail; earlier sections can be non-monotonic.
    k = len(grid) - 1
    while k > 0 and grid[k - 1][1] >= grid[k][1]:
        k -= 1
    grid = grid[k:]
    sigmas = [s for s, _ in grid]

    lo_min, e0 = sigmas[0], float("inf")
    for s in sigmas:
        e0 = eps_ss(s)
        if e0 != float("inf"):
            lo_min = s
            break

    if target > e0:
        return round(lo_min, 2), e0

    within = [s for s, e in grid if e <= target]
    seed = min(within) if within else sigmas[-1]
    idx = sigmas.index(seed)
    a = sigmas[idx - 1] if idx > 0 else seed
    b = seed

    ea, eb = eps_ss(a), eps_ss(b)
    best = min([(a, ea), (b, eb)], key=lambda se: abs(se[1] - target))

    steps = 0
    while eb > target and steps < 25:
        a, ea = b, eb
        b *= 1.4
        eb = eps_ss(b); steps += 1
        if abs(eb - target) < abs(best[1] - target):
            best = (b, eb)
    while ea < target and a > lo_min and steps < 40:
        b, eb = a, ea
        a = max(lo_min, a * 0.7)
        ea = eps_ss(a); steps += 1
        if abs(ea - target) < abs(best[1] - target):
            best = (a, ea)

    for _ in range(max_iter):
        if abs(best[1] - target) <= tol:
            break
        if (b - a) <= 1e-3 * b:
            break
        mid = 0.5 * (a + b)
        em = eps_ss(mid)
        if abs(em - target) < abs(best[1] - target):
            best = (mid, em)
        if em > target:
            a = mid
        else:
            b = mid
    return round(best[0], 2), best[1]


def train_student(dataset, teachers_preds, sigma, stdnt_data,
                  eval_data, eval_labels, ckpt):
    """Relabels the queries at this sigma, trains the student, returns accuracy."""
    labels, _, _ = aggregation.noisy_max(
        teachers_preds, float(sigma), return_clean_votes=True)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    deep_cnn.train(dataset, stdnt_data, label_tensor, ckpt,
                   epochs=STUDENT_EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    preds = deep_cnn.softmax_preds(dataset, eval_data, ckpt)
    return metrics.accuracy(preds, eval_labels.numpy())


def write_md(results, ts, te):
    """(Re)writes the best-per-epsilon markdown for one teacher-epoch value."""
    lines = [
        "# PATE noise sweep — best noise scale per target ε",
        "",
        f"_Generated {ts}; δ={DELTA}, {STUDENT_QUERIES} queries, "
        f"student {STUDENT_EPOCHS} epochs, teacher {te} epochs._",
        "",
        f"For each target ε, σ is bisected so the data-dependent ε lands "
        f"within ±{EPS_TOL} of the target.",
        "",
        "| Dataset | Teachers | Target ε | Chosen σ | Achieved ε | Student acc |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['nb_teachers']} | {r['target_eps']:.0f} | "
            f"{r['sigma']:.1f} | {r['achieved_eps']:.2f} | {r['accuracy']:.4f} |")
    with open(md_path(te), "w") as f:
        f.write("\n".join(lines) + "\n")


def _summary_header():
    return ("dataset\tnb_teachers\tteacher_epochs\ttarget_eps\tsigma\t"
            "achieved_eps\taccuracy\twall_time_s\tstatus\n")


def _student_split(dataset):
    test_data, test_labels = LD[dataset](test_only=True, data_dir=DATA_DIR)
    return (
        test_data[:STUDENT_QUERIES],
        test_data[STUDENT_QUERIES:],
        test_labels[STUDENT_QUERIES:],
    )


def _write_eps_sweep(run_dir, cfg, pairs):
    with open(os.path.join(run_dir, f"{cfg}_eps_sweep.json"), "w") as jf:
        json.dump([{"sigma": s, "epsilon": e} for s, e in pairs], jf, indent=2)


def _append_result(summary, dataset, nb, te, target, sigma, achieved, acc, dt):
    with open(summary, "a") as s:
        s.write(f"{dataset}\t{nb}\t{te}\t{target:.0f}\t{sigma}\t"
                f"{achieved:.4f}\t{acc:.4f}\t{dt:.2f}\tOK\n")


def _append_failure(summary, dataset, nb, te, wall_time, exc):
    with open(summary, "a") as s:
        s.write(f"{dataset}\t{nb}\t{te}\t-\t-\t-\t-\t"
                f"{wall_time:.2f}\tFAIL:{exc}\n")


def _run_config(dataset, nb, te, run_dir, summary, results, ts):
    cfg = f"{dataset}_t{nb}_e{te}"
    train_dir = os.path.join(TEACHER_BASE, cfg)
    log("=" * 60)
    log(f"CONFIG {cfg}")
    log("=" * 60)

    t_cfg = time.time()
    try:
        log("  training teachers ...")
        train_teachers(dataset, nb, train_dir, te)

        log("  querying ensemble (once) ...")
        stdnt_data, eval_data, eval_labels = _student_split(dataset)
        teachers_preds = ensemble_preds(dataset, nb, stdnt_data, train_dir)
        _, clean_votes, _ = aggregation.noisy_max(
            teachers_preds, 1.0, return_clean_votes=True)

        log("  sweeping sigma -> epsilon (cheap) ...")
        pairs = sweep_epsilon(clean_votes)
        _write_eps_sweep(run_dir, cfg, pairs)
        log("    " + ", ".join(f"σ{int(s)}→ε{e:.1f}" for s, e in pairs))

        chosen = {}
        for target in TARGET_EPS:
            sigma, achieved, fast = solve_sigma(clean_votes, pairs, target)
            if sigma not in chosen:
                path = "data-indep" if fast else "bisection"
                log(f"  target ε={target:.0f}: σ={sigma} (ε≈{achieved:.2f})"
                    f" [{path}] -> training student ...")
                t0 = time.time()
                ckpt = os.path.join(train_dir, f"student_sigma{sigma}.pt")
                acc = train_student(dataset, teachers_preds, sigma,
                                    stdnt_data, eval_data, eval_labels, ckpt)
                chosen[sigma] = (achieved, acc, time.time() - t0)
                log(f"    student accuracy: {acc:.4f}")

            achieved, acc, dt = chosen[sigma]
            results.append({
                "dataset": dataset, "nb_teachers": nb,
                "target_eps": target, "sigma": sigma,
                "achieved_eps": achieved, "accuracy": acc,
            })
            _append_result(summary, dataset, nb, te, target, sigma, achieved,
                           acc, dt)
            write_md(results, ts, te)
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        log("  CONFIG FAILED:\n" + traceback.format_exc())
        _append_failure(summary, dataset, nb, te, time.time() - t_cfg, exc)

    log(f"  config done in {time.time() - t_cfg:.1f}s\n")


def _run_all_sweeps(run_dir, summary, ts):
    for te in TEACHER_EPOCHS:
        results = []
        for dataset in DATASETS:
            for nb in NB_TEACHERS:
                _run_config(dataset, nb, te, run_dir, summary, results, ts)


def _apply_args(args):
    global DATASETS, NB_TEACHERS, TEACHER_EPOCHS, STUDENT_EPOCHS
    global STUDENT_QUERIES, SIGMA_GRID

    if args.datasets:
        DATASETS = [d.strip() for d in args.datasets.split(",")]
    if args.teachers:
        NB_TEACHERS = [int(t) for t in args.teachers.split(",")]
    if args.epochs:
        TEACHER_EPOCHS = [int(e) for e in args.epochs.split(",")]
    if args.queries:
        STUDENT_QUERIES = args.queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated subset of datasets to run.")
    parser.add_argument("--teachers", default=None,
                        help="Comma-separated subset of teacher counts to run.")
    parser.add_argument("--epochs", default=None,
                        help="Comma-separated teacher-epoch values; one full "
                             "sweep (-> its own _e<te>.md) is run per value.")
    parser.add_argument("--queries", type=int, default=None,
                        help="Number of student queries (default 1000).")
    args = parser.parse_args()
    _apply_args(args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    qtag = "" if STUDENT_QUERIES == 1000 else f"_q{STUDENT_QUERIES}"
    run_dir = os.path.join(PROJECT_DIR, "logs", f"pate_sweep_{ts}{qtag}")
    os.makedirs(run_dir, exist_ok=True)
    summary = os.path.join(run_dir, "summary.tsv")
    with open(summary, "w") as s:
        s.write(_summary_header())

    os.makedirs(DATA_DIR, exist_ok=True)
    log(f"device: {deep_cnn.get_device()}")
    log(f"run dir: {run_dir}")
    log(f"teacher epochs: {TEACHER_EPOCHS} | queries: {STUDENT_QUERIES}")
    log(f"markdown: {', '.join(md_path(te) for te in TEACHER_EPOCHS)}\n")

    _run_all_sweeps(run_dir, summary, ts)

    log("ALL DONE.")
    log(f"  best-per-ε tables: "
        f"{', '.join(md_path(te) for te in TEACHER_EPOCHS)}")
    log(f"  details: {run_dir}")


if __name__ == "__main__":
    main()
