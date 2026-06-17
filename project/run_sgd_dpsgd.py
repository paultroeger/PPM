"""
Batch runner for SGD and DP-SGD experiments.

Logs are written under project/logs with one log file, one metrics JSON, and a
summary.tsv per sweep.

Note: Claude was used to help create these multi-training scripts.
"""

import json
import os
import time
import traceback
from datetime import datetime

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from test_load_data import get_mnist, get_svhn, get_cifar10
from deep_cnn import deep_cnn, get_device

DATASETS = ["MNIST", "SVHN", "CIFAR10"]
EPOCHS = [10, 50, 100]
EPSILONS = [1.0, 4.0, 8.0]
BATCH_SIZE = 64
STEP_SIZE = 0.01
MOMENTUM = 0.9
C = 1.0
DELTA = 1e-5
USE_SIMPLE = False  # --model simple uses the MLP baseline

LOADERS = {"MNIST": get_mnist, "SVHN": get_svhn, "CIFAR10": get_cifar10}

cross_entropy_loss = nn.CrossEntropyLoss()
device = get_device()


def make_logger(path):
    """Returns a (log, file) pair; log() writes to both stdout and the file."""
    f = open(path, "w")

    def log(msg=""):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    return log, f


def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_run(dataset, epochs, private, log, epsilon=None):
    """Trains one model and returns a metrics dict, logging each epoch."""
    train_loader, test_loader = LOADERS[dataset](BATCH_SIZE)
    model = deep_cnn(dataset, simple=USE_SIMPLE).to(device)

    privacy_engine = None
    if private:
        # Opacus swaps unsupported layers such as BatchNorm when needed.
        model = ModuleValidator.fix(model).to(device)
        optimizer = torch.optim.SGD(model.parameters(), STEP_SIZE, momentum=MOMENTUM)
        privacy_engine = PrivacyEngine()
        privacy_engine.accountant.DEFAULT_ALPHAS = list(range(2, 512))
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=epsilon,
            target_delta=DELTA,
            epochs=epochs,
            max_grad_norm=C,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), STEP_SIZE, momentum=MOMENTUM)

    acc = 0.0
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        last_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = cross_entropy_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        acc = evaluate(model, test_loader)
        best_acc = max(best_acc, acc)
        if private:
            eps_spent = privacy_engine.get_epsilon(DELTA)
            log(f"  Epoch {epoch + 1}/{epochs} — loss: {last_loss:.4f} | "
                f"accuracy: {acc:.4f} | ε = {eps_spent:.2f}, δ = {DELTA}")
        else:
            log(f"  Epoch {epoch + 1}/{epochs} — loss: {last_loss:.4f} | "
                f"accuracy: {acc:.4f}")

    metrics = {
        "method": "dpsgd" if private else "sgd",
        "model": "simple" if USE_SIMPLE else "cnn",
        "dataset": dataset.lower(),
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "lr": STEP_SIZE,
        "momentum": MOMENTUM,
        "final_accuracy": acc,
        "best_accuracy": best_acc,
        "device": str(device),
    }
    if private:
        metrics.update({
            "target_epsilon": epsilon,
            "achieved_epsilon": privacy_engine.get_epsilon(DELTA),
            "delta": DELTA,
            "max_grad_norm": C,
        })
    return metrics


def _sweep_cases(private):
    for dataset in DATASETS:
        for epochs in EPOCHS:
            if private:
                for epsilon in EPSILONS:
                    yield dataset, epochs, epsilon
            else:
                yield dataset, epochs, None


def _run_tag(dataset, epochs, epsilon):
    tag = f"{dataset.lower()}_e{epochs}"
    if epsilon is not None:
        tag += f"_eps{int(epsilon)}"
    return tag


def _summary_header(private):
    if private:
        return ("dataset\tepochs\tepsilon_target\tepsilon_achieved\tbatch_size\t"
                "lr\tfinal_accuracy\tbest_accuracy\twall_time_s\tstatus\tlog_file\n")
    return ("dataset\tepochs\tbatch_size\tlr\tfinal_accuracy\tbest_accuracy\t"
            "wall_time_s\tstatus\tlog_file\n")


def _log_header(log, dataset, epochs, private, epsilon):
    log("=" * 60)
    if private:
        log(f"DP-SGD | {dataset} | epochs={epochs} | target ε={epsilon} | "
            f"batch={BATCH_SIZE} | lr={STEP_SIZE} | C={C} | δ={DELTA}")
    else:
        log(f"SGD | {dataset} | epochs={epochs} | batch={BATCH_SIZE} | "
            f"lr={STEP_SIZE}")
    log("=" * 60)


def _summary_row(dataset, epochs, epsilon, metrics, wall_time, status, log_file):
    tail = f"{wall_time:.2f}\t{status}\t{log_file}\n"
    if metrics is None:
        if epsilon is None:
            return (f"{dataset.lower()}\t{epochs}\t{BATCH_SIZE}\t{STEP_SIZE}\t"
                    f"-\t-\t{tail}")
        return (f"{dataset.lower()}\t{epochs}\t{epsilon}\t-\t{BATCH_SIZE}\t"
                f"{STEP_SIZE}\t-\t-\t{tail}")

    if epsilon is None:
        return (f"{dataset.lower()}\t{epochs}\t{BATCH_SIZE}\t{STEP_SIZE}\t"
                f"{metrics['final_accuracy']:.4f}\t{metrics['best_accuracy']:.4f}\t"
                f"{tail}")
    return (f"{dataset.lower()}\t{epochs}\t{epsilon}\t"
            f"{metrics['achieved_epsilon']:.4f}\t{BATCH_SIZE}\t{STEP_SIZE}\t"
            f"{metrics['final_accuracy']:.4f}\t{metrics['best_accuracy']:.4f}\t"
            f"{tail}")


def _run_sweep(run_dir, private):
    method = "DP-SGD" if private else "SGD"
    summary = os.path.join(run_dir, "summary.tsv")
    with open(summary, "w") as s:
        s.write(_summary_header(private))

    for dataset, epochs, epsilon in _sweep_cases(private):
        tag = _run_tag(dataset, epochs, epsilon)
        log_file = f"{tag}.log"
        log, f = make_logger(os.path.join(run_dir, log_file))
        _log_header(log, dataset, epochs, private, epsilon)

        t0 = time.time()
        try:
            metrics = train_run(dataset, epochs, private=private, log=log,
                                epsilon=epsilon)
            wall_time = time.time() - t0
            metrics["wall_time_s"] = round(wall_time, 2)
            with open(os.path.join(run_dir, f"{tag}_metrics.json"), "w") as mf:
                json.dump(metrics, mf, indent=2, sort_keys=True)
            status = "OK"
        except Exception:
            wall_time = time.time() - t0
            metrics = None
            status = "FAIL"
            log("FAILED:\n" + traceback.format_exc())
        finally:
            f.close()

        row = _summary_row(dataset, epochs, epsilon, metrics, wall_time, status,
                           log_file)
        with open(summary, "a") as s:
            s.write(row)
        print(f"[{method}] {tag}: {status} ({wall_time:.1f}s)")


def run_sgd(run_dir):
    return _run_sweep(run_dir, private=False)


def run_dpsgd(run_dir):
    return _run_sweep(run_dir, private=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SGD / DP-SGD benchmark runner")
    parser.add_argument("--only", choices=("sgd", "dpsgd", "both"),
                        default="both")
    parser.add_argument("--model", choices=("cnn", "simple", "both"), default="cnn")
    args = parser.parse_args()

    models = ("cnn", "simple") if args.model == "both" else (args.model,)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    global USE_SIMPLE
    for model in models:
        USE_SIMPLE = (model == "simple")
        tag = "_simple" if USE_SIMPLE else ""

        print(f"Device: {device} | model: {model}")
        if args.only in ("sgd", "both"):
            sgd_dir = os.path.join(base, f"sgd{tag}_{ts}")
            os.makedirs(sgd_dir, exist_ok=True)
            print(f"=== SGD sweep ({model}) ===")
            run_sgd(sgd_dir)
            print(f"SGD logs: {sgd_dir}")
        if args.only in ("dpsgd", "both"):
            dpsgd_dir = os.path.join(base, f"dpsgd{tag}_{ts}")
            os.makedirs(dpsgd_dir, exist_ok=True)
            print(f"=== DP-SGD sweep ({model}) ===")
            run_dpsgd(dpsgd_dir)
            print(f"DP-SGD logs: {dpsgd_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
