"""
Data loader for MNIST, SVHN, and CIFAR-10 datasets.
Downloads automatically on first run into ./data/
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(f"MNIST: {len(train)} train, {len(test)} test samples")
    return train_loader, test_loader


def get_svhn(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))
    ])
    train = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(f"SVHN: {len(train)} train, {len(test)} test samples")
    return train_loader, test_loader


def get_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
    ])
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(f"CIFAR-10: {len(train)} train, {len(test)} test samples")
    return train_loader, test_loader


def get_all_datasets(batch_size=64):
    print("Loading all datasets...")
    mnist_train, mnist_test = get_mnist(batch_size)
    svhn_train, svhn_test = get_svhn(batch_size)
    cifar_train, cifar_test = get_cifar10(batch_size)
    print("All datasets loaded.")
    return {
        'mnist': (mnist_train, mnist_test),
        'svhn':  (svhn_train, svhn_test),
        'cifar10': (cifar_train, cifar_test),
    }


if __name__ == "__main__":
    datasets_dict = get_all_datasets()