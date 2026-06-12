"""
Data loader for MNIST, SVHN, and CIFAR-10 datasets.
Downloads automatically on first run into ./data/

Besides the DataLoader-based functions (get_mnist, get_svhn, get_cifar10) used
by sgd_vs_dpsgd.py, this file also provides the tensor-based API that the
PATE scripts need (ld_mnist, ld_svhn, ld_cifar10, partition_dataset). These
mirror the functions of the same name in pate_2017/input.py: PATE has to slice
the training set into disjoint partitions (one per teacher) and to relabel a
slice of the test set with teacher votes, which is much easier on raw tensors
than on DataLoaders.
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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(f"CIFAR-10: {len(train)} train, {len(test)} test samples")
    return train_loader, test_loader


# ── Input for PATE (mirrors pate_2017/input.py) ──


def _dataset_to_tensors(dataset):
    """
    Materializes a torchvision dataset (with its transform applied) into two
    tensors: data of shape (N, C, H, W) and labels of shape (N,).
    """
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    data_batches, label_batches = [], []
    for x, y in loader:
        data_batches.append(x)
        label_batches.append(y)
    return torch.cat(data_batches), torch.cat(label_batches)


def ld_mnist(test_only=False, data_dir='./data'):
    """
    Loads MNIST as tensors. Returns (test_data, test_labels) if test_only,
    otherwise (train_data, train_labels, test_data, test_labels).
    """
    # same normalization as get_mnist so that the PATE teachers/student see
    # exactly the same preprocessed inputs as sgd_vs_dpsgd
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    test_data, test_labels = _dataset_to_tensors(test)
    if test_only:
        return test_data, test_labels
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_data, train_labels = _dataset_to_tensors(train)
    return train_data, train_labels, test_data, test_labels


def ld_svhn(test_only=False, data_dir='./data'):
    """Same as ld_mnist but for SVHN."""
    # same normalization as get_svhn
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    test_data, test_labels = _dataset_to_tensors(test)
    if test_only:
        return test_data, test_labels
    train = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    train_data, train_labels = _dataset_to_tensors(train)
    return train_data, train_labels, test_data, test_labels


def ld_cifar10(test_only=False, data_dir='./data'):
    """Same as ld_mnist but for CIFAR-10."""
    # same normalization as get_cifar10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_data, test_labels = _dataset_to_tensors(test)
    if test_only:
        return test_data, test_labels
    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_data, train_labels = _dataset_to_tensors(train)
    return train_data, train_labels, test_data, test_labels


def partition_dataset(data, labels, nb_teachers, teacher_id):
    """
    Simple partitioning algorithm that returns the right portion of the data
    needed by a given teacher out of a certain nb of teachers.
    Identical to pate_2017/input.partition_dataset.
    """
    assert len(data) == len(labels)
    assert int(teacher_id) < int(nb_teachers)

    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)

    # Compute start, end indices of partition
    start = teacher_id * batch_len
    end = (teacher_id + 1) * batch_len

    return data[start:end], labels[start:end]


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