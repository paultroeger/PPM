"""
compute_norm_stats.py

Computes per-channel normalization statistics (mean, std) for MNIST, SVHN,
and CIFAR-10, directly from each dataset's TRAINING split.

- Statistics are computed on the train split only (never test), which is the
  correct practice: the same train-derived numbers are then applied to both
  train and test at training time.
- Uses a global single-pass accumulation (sum of x and sum of x^2 over every
  pixel), so the std is the true population std -- not the slightly-wrong value
  you get by averaging per-image stds.

Usage:
    python compute_norm_stats.py
    python compute_norm_stats.py --datasets mnist cifar10
    python compute_norm_stats.py --data-root ./data --batch-size 1000

Requires: torch, torchvision  (SVHN also needs scipy to read its .mat files)
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = './data'

# Dataset registry: name -> (loader builder, number of channels)
def _mnist():
    return datasets.MNIST(root=ROOT, train=True, download=True,
                          transform=transforms.ToTensor())

def _svhn():
    return datasets.SVHN(root=ROOT, split='train', download=True,
                         transform=transforms.ToTensor())

def _cifar10():
    return datasets.CIFAR10(root=ROOT, train=True, download=True,
                            transform=transforms.ToTensor())

DATASETS = {
    'mnist':   (_mnist,   1),
    'svhn':    (_svhn,    3),
    'cifar10': (_cifar10, 3),
}


def compute_mean_std(dataset, channels, batch_size=1000, num_workers=2):
    """Per-channel mean and std over a dataset already scaled to [0,1].

    mean = E[x]
    std  = sqrt(E[x^2] - E[x]^2)
    Accumulated globally over all pixels (correct population std).
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(channels, dtype=torch.float64) # sum of all pixel values, per channel
    channel_sq_sum = torch.zeros(channels, dtype=torch.float64) # sum of squared values, per channel
    n_pixels = 0 # total pixel count

    for imgs, _ in loader: #iterate the whole dataset batch by batch, imgs is a tensor of shape (B, C, H, W) where B is the batch size, C is the number of channels, H and W are the height and width of the images:
    # MNIST:         (B, 1, 28, 28) - grayscale ( 1 colour channel), 28×28
    # SVHN/CIFAR-10: (B, 3, 32, 32) - RGB (3 colour channels), 32×32
        b, c, h, w = imgs.shape
        imgs = imgs.double().view(b, c, -1)          # (B, C, H*W)
        channel_sum += imgs.sum(dim=[0, 2])
        channel_sq_sum += (imgs ** 2).sum(dim=[0, 2])
        n_pixels += b * h * w

    mean = channel_sum / n_pixels # E[x] = sum(x) / n empirical mean approximates expected value
    var = channel_sq_sum / n_pixels - mean ** 2 # E[x^2] - E[x]^2 = (sum(x^2) / n) - (sum(x) / n)^2 empirical variance approximates population variance
    std = torch.sqrt(var.clamp(min=0))               # clamp guards tiny negatives
    return mean.tolist(), std.tolist() # return the results


def fmt(values):
    return "(" + ", ".join(f"{v:.4f}" for v in values) + ")"


def main():
    for name, (builder, channels) in DATASETS.items():
        print(f"\n[{name}] loading...")
        ds = builder()
        mean, std = compute_mean_std(ds, channels)
        print(f"  mean = {fmt(mean)}")
        print(f"  std  = {fmt(std)}")
        print(f"  -> transforms.Normalize({fmt(mean)}, {fmt(std)})")


if __name__ == "__main__":
    main()