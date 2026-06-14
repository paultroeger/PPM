import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    if torch.cuda.is_available():
        print("cuda")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("mps")
        return torch.device("mps")
    print("cpu")
    return torch.device("cpu")


def deep_cnn(dataset):
    if dataset == "MNIST":
        # MNIST: simplest, grayscale 28x28, easy to classify
        # 2 conv layers, small filters
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #2d convolutional layer with kernel of size 3, 1 input channel, 32 output channels
                                                        # adding padding = 1 to have a boundary and not shrink the 
            nn.ReLU(),                                  # relu activation function
            nn.MaxPool2d(2),                            # MaxPool2d(2) MaxPool takes the maximum value of each window, 2 stands for window size
                                                        # -> effectively turns a 28x28px picture into a 14x14px picture
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#32 input channels from above now producing 64 output channels
            nn.ReLU(),
            nn.MaxPool2d(2),                            # second max pool turns 14x14px picture into a 7x7pc picture
            nn.Flatten(),                               # flattens the tensors of dimension 256x64x7x7 into a 
                                                        # matrix of dimension IR^((64*7*7) x256)
            nn.Linear(64 * 7 * 7, 256),                 # applies the linear layer that we flattened the tensor for
            nn.ReLU(),                                  # relu activation function
            nn.Dropout(0.5),                            # dropout layer randomly sets some of the neurons to zero; prevents overfitting
            nn.Linear(256, 10)                          # another linear layer
        )
    elif dataset == "SVHN":
        # SVHN: harder, color 32x32, real-world street numbers with clutter
        # 3 conv layers, more filters, needs more capacity than MNIST
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    elif dataset == "CIFAR10":
        # CIFAR-10: hardest, color 32x32, diverse object categories
        # 3-4 conv layers, even more filters, needs BatchNorm to train stably
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    # Simple MLP baseline - fewer parameters can outperform CNN under strong DP constraints because:    
    #   more parameters 
    #=> more epsilon budget
    #=> lower accuracy
    elif dataset == "SIMPLE": 
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    return model


# ── Training for PATE (mirrors pate_2017/deep_cnn.py) ──
#
# The TF1 version had train(images, labels, ckpt_path) and
# softmax_preds(images, ckpt_path). We keep the same shape, so the PATE
# teacher/student scripts translate almost one-to-one.
# The training loop is the plain (non-private) SGD loop from sgd_vs_dpsgd.py:
# PATE gets its privacy from the noisy aggregation of teacher votes, NOT from
# the training algorithm, so teachers and student train with normal SGD.


def train(dataset, data, labels, ckpt_path, epochs=10, batch_size=64, lr=0.01,
          momentum=0.9):
    """Trains a deep_cnn model on the given tensors with normal SGD and saves the
    final weights to ckpt_path."""
    device = get_device()
    model = deep_cnn(dataset.upper()).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum)

    train_set = TensorDataset(data, labels.long())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # step 1: compute loss
            output = model(x_batch)
            loss = cross_entropy_loss(output, y_batch)

            # step 2: compute gradients, step 3: update parameters
            loss.backward()
            optimizer.step()

        print(f"  Epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

    # save the trained weights so teachers can be queried later by the student
    torch.save(model.state_dict(), ckpt_path)

    return True


def softmax_preds(dataset, data, ckpt_path, batch_size=256):
    """Loads the weights stored at ckpt_path and computes softmax predictions."""
    device = get_device()
    model = deep_cnn(dataset.upper()).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            output = model(data[i:i + batch_size].to(device))
            preds.append(torch.softmax(output, dim=1).cpu().numpy())

    return np.concatenate(preds)