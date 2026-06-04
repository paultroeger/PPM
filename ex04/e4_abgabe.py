# %% Imports and useful functions
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pytest

sns.set_style("whitegrid")


# %%

# This is the gradient of the logistic loss
def gradient(weights, xi, yi):
    #yi = -1 if yi == 0 else 1
    exponent = yi * (xi.dot(weights))
    return - (yi * xi) / (1 + np.exp(exponent))


def rdpgd(iterations, alpha, epsilon_bar):
    X_train, X_test, y_train, y_test = prepare_data(load_csv())

    weights = np.zeros(X_train.shape[1])
    n = X_train.shape[0]
    # This needs to be done, as otherwise the gradient is always 0 when somebody lived
    y_train = np.where(y_train == 1, 1, -1)

    # everytime we add noise on the gradient we need privacy budget
    # using sequential composition we split epsilon
    C = 1.0 # Maybe change this
    epsilon_step = epsilon_bar / iterations
    sigma = np.sqrt(alpha / (2 * epsilon_step))

    for t in range(iterations):

        # gradient
        grad_sum = np.zeros_like(weights)
        for i in range(n):
            grad = gradient(weights, X_train[i], y_train[i])

            # do we clip here?
            grad_norm = np.linalg.norm(grad)
            if grad_norm > C:
                grad = grad * (C / grad_norm)

            grad_sum += grad

        # noise
        noise = np.random.normal(loc=0, scale=sigma, size=weights.shape)

        # weight update
        learning_rate = 0.1
        weights -= learning_rate * (grad_sum + noise) / n

    model = LogisticRegression(max_iter=100)
    model.coef_ = weights.reshape(1, -1)
    model.intercept_ = np.array([0.0])

    model.classes_ = np.array([0, 1])

    return model.score(X_test, y_test)


def dpsgd(iterations, epsilon, delta, learning_rate, batch_size, C):
    X_train, X_test, y_train, y_test = prepare_data(load_csv())

    n = X_train.shape[0]
    weights = np.zeros(X_train.shape[1])
    # This needs to be done, as otherwise the gradient is always 0 when somebody lived
    y_train = np.where(y_train == 1, 1, -1)

    # This is the calculation of the moments accountant (see page 26 in the slides)
    q = batch_size / n
    epsilon_step = epsilon / (q * np.sqrt(iterations))
    sigma = C * np.sqrt(2 * np.log(1.25 / delta)) / epsilon_step

    for t in range(iterations):
        indices = np.random.choice(n, size=batch_size, replace=False)

        grad_sum = np.zeros_like(weights)
        for i in indices:
            g = gradient(weights, X_train[i], y_train[i])

            g_norm = np.linalg.norm(g)
            if g_norm > C:
                g = g * (C / g_norm)

            grad_sum += g

        noise = np.random.normal(0, sigma, size=weights.shape)
        weights -= learning_rate * (grad_sum + noise) / batch_size

    model = LogisticRegression(max_iter=100)
    model.coef_ = weights.reshape(1, -1)
    model.intercept_ = np.array([0.0])
    model.classes_ = np.array([0, 1])

    return model.score(X_test, y_test)


# Our functions
def load_csv():
    return pd.read_csv("./covid19-data-kaggle/covid.csv")


def prepare_data(df):
    df = df.drop(columns=["id"])
    df = df.replace([97, 98, 99], float('nan'))
    df = df.replace([2, 3], [0, 2])
    # days since
    reference_date = pd.Timestamp('2020-01-01')

    df['entry_date'] = pd.to_datetime(df['entry_date'], format="%d-%m-%Y")
    df['date_symptoms'] = pd.to_datetime(df['date_symptoms'], format="%d-%m-%Y")

    df['entry_date'] = (df['entry_date'] - reference_date).dt.days
    df['date_symptoms'] = (df['date_symptoms'] - reference_date).dt.days

    # still alive or dead
    df['date_died'] = np.where(df['date_died'] == '9999-99-99', 0, 1)

    # X,y
    X = df.drop('date_died', axis=1)
    y = df['date_died']

    # imputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # add random state

    # Standardize all rows so that model does not predict only survivors
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('DPSGD Privacy / Utility Trade-off', fontsize=14, fontweight='bold')

    configs = [
        ('epsilon', 'epsilon', 'Accuracy vs Privacy Budget'),
        ('iterations', 'Iterations', 'Accuracy vs Iterations'),
        ('learning_rate', 'Learning Rate', 'Accuracy vs Learning Rate'),
        ('batch_size', 'Batch Size', 'Accuracy vs Batch Size'),
    ]

    for ax, (key, xlabel, title) in zip(axes.flat, configs):
        x, y = results[key]
        ax.plot(x, y, marker='o', linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Test Accuracy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key == 'learning_rate':
            ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('dpsgd_tradeoff.png', dpi=150)
    plt.show()


def plot_task4_results(results_dpsgd, results_rdp):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('RDPGD vs DPSGD Performance Comparison', fontsize=14, fontweight='bold')

    x_dpsgd, y_dpsgd = results_dpsgd['iterations']
    x_rdp, y_rdp = results_rdp['iterations']
    axes[0, 0].plot(x_dpsgd, y_dpsgd, marker='o', linewidth=2, label='DPSGD')
    axes[0, 0].plot(x_rdp, y_rdp, marker='o', linewidth=2, label='RDPGD')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy vs Iterations')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    x_dpsgd, y_dpsgd = results_dpsgd['iterations_time']
    x_rdp, y_rdp = results_rdp['iterations_time']
    axes[0, 1].plot(x_dpsgd, y_dpsgd, marker='o', linewidth=2, label='DPSGD')
    axes[0, 1].plot(x_rdp, y_rdp, marker='o', linewidth=2, label='RDPGD')
    axes[0, 1].set_xlabel('Iterations')
    axes[0, 1].set_ylabel('Runtime (s)')
    axes[0, 1].set_title('Runtime vs Iterations')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    x, y = results_dpsgd['batch_size']
    axes[1, 0].plot(x, y, marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('DPSGD Accuracy vs Batch Size')
    axes[1, 0].grid(True, alpha=0.3)

    x, y = results_dpsgd['batch_size_time']
    axes[1, 1].plot(x, y, marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Runtime (s)')
    axes[1, 1].set_title('DPSGD Runtime vs Batch Size')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task4_performance.png', dpi=150)
    plt.show()


# %% Tasks
def task1():
    print('-' * 6, 'Task 1', '-' * 6)
    df = load_csv()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    print("Non-Private Accuracy:", model.score(X_test, y_test))

    print()

# 1: In each gradient descent iteration we spend epsilon_bar / iterations privacy budget by adding
# noise to the average gradient. Through sequential composition this adds up per iteration, so
# after all iterations are done we have a total cost of epsilon_bar.
#
# 2: RDPGD has lower accuracy (0.7 vs 0.94) than the non-private logistic regression, as we add noise to protect privacy.
# This, however, makes the gradient less precise, so there is a privacy-utility trade-off.
def task2():
    print('-' * 6, 'Task 2', '-' * 6)

    print("RDP Accuracy:", rdpgd(100, 15.0, 1.0))

    print()

# 1: First, we can see that iterations have by far the largest impact on accuracy, with 500 being the best overall.
# Learning rate also seems to have a high impact, however, if a learning rate above 0.1 is chosen, the accuracy
# fluctuates a lot, thus leading to inconsistent results. A learning rate of 0.1 seems like a good compromise.
# Both batch size and epsilon seem to have an overall smaller impact, with epsilon=1 and batch_size=32 seeming good.
#
# 2: We can recommend using a large number of iterations, with 500 performing the best overall. A learning rate of
# 0.1 with epsilon=1 and batch_size=32 are reasonable choices.
def task3():
    print('-' * 6, 'Task 3', '-' * 6)

    # Optimal values, higher learning rate might increase accuracy, but accuracy fluctuates a lot then
    # Total privacy cost of 1
    print("DPS Accuracy:", dpsgd(500, 1.0, 1e-5, 0.1, 32, 4.0))

    results = {}

    # default values
    it_def, eps_def, delta_def, lr_def, bs_def, c_def = 100, 1.0, 1e-5, 0.1, 32, 4.0

    # epsilon
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    acc_eps = []
    for eps in epsilons:
        score = dpsgd(it_def, eps, delta_def,
                      lr_def, bs_def, c_def)
        acc_eps.append(score)
    results['epsilon'] = (epsilons, acc_eps)

    # iterations
    iters = [10, 50, 100, 200, 500]
    acc_iter = []
    for it in iters:
        score = dpsgd(it, eps_def, delta_def,
                      lr_def, bs_def, c_def)
        acc_iter.append(score)
    results['iterations'] = (iters, acc_iter)

    # lr
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    acc_lr = []
    for lr in lrs:
        score = dpsgd(it_def, eps_def, delta_def,
                      lr, bs_def, c_def)
        acc_lr.append(score)
    results['learning_rate'] = (lrs, acc_lr)

    # batch_size
    batch_sizes = [8, 16, 32, 64, 128]
    acc_bs = []
    for bs in batch_sizes:
        score = dpsgd(it_def, eps_def, delta_def,
                      lr_def, bs, c_def)
        acc_bs.append(score)
    results['batch_size'] = (batch_sizes, acc_bs)

    plot_results(results)

    print()


# 1: The plot shows that increasing the number of iterations improves the utility of both algorithms, but for DPSGD
# the effect is much stronger, while RDPGD only improves very minimally across iterations. Additionally, the runtime
# increases very significantly for RDPGD, while DPGSD barely has any increase.
# For DPSGD, batch size has a smaller and less significant effect on accuracy, with the best result in being at
# batch size 16. Runtime increases very slightly with larger batch sizes.
#
# 2: The main reason for the runtime difference is that RDPGD always uses the full training data for gradient descent.
# DPSGD, on the other hand, only uses a batch of the training data, so each iteration is much cheaper and the total
# runtime is much lower.
def task4():
    print('-' * 6, 'Task 4', '-' * 6)

    results_rdp = {}
    results_dpsgd = {}

    epsilon = 1.0
    delta = 1e-5
    alpha = 15

    # According to this, alpha needs to be larger than 12.51, otherwise epsilon_bar is a negative value
    epsilon_bar = epsilon - np.log(1 / delta) / (alpha - 1)

    # default values
    it_def, lr_def, bs_def, c_def = 500, 0.1, 32, 4.0

    # iterations
    dpsgd_iters = [10, 50, 100, 200, 500]
    rdpgd_iters = [10, 50, 100]
    acc_iter_dpsgd = []
    acc_iter_rdp = []
    time_iter_dpsgd = []
    time_iter_rdp = []

    for it in dpsgd_iters:
        start = dt.datetime.now()
        score = dpsgd(it, epsilon, delta, lr_def, bs_def, c_def)
        runtime = (dt.datetime.now() - start).total_seconds()
        acc_iter_dpsgd.append(score)
        time_iter_dpsgd.append(runtime)

    for it in rdpgd_iters:
        start = dt.datetime.now()
        score = rdpgd(it, alpha, epsilon_bar)
        runtime = (dt.datetime.now() - start).total_seconds()
        acc_iter_rdp.append(score)
        time_iter_rdp.append(runtime)

    results_dpsgd['iterations'] = (dpsgd_iters, acc_iter_dpsgd)
    results_rdp['iterations'] = (rdpgd_iters, acc_iter_rdp)
    results_dpsgd['iterations_time'] = (dpsgd_iters, time_iter_dpsgd)
    results_rdp['iterations_time'] = (rdpgd_iters, time_iter_rdp)

    # batch_size
    batch_sizes = [8, 16, 32, 64, 128, 256]
    acc_bs_dpsgd = []
    time_bs_dpsgd = []
    for bs in batch_sizes:
        start = dt.datetime.now()
        score = dpsgd(it_def, epsilon, delta,
                      lr_def, bs, c_def)
        runtime = (dt.datetime.now() - start).total_seconds()
        acc_bs_dpsgd.append(score)
        time_bs_dpsgd.append(runtime)

    results_dpsgd['batch_size'] = (batch_sizes, acc_bs_dpsgd)
    results_dpsgd['batch_size_time'] = (batch_sizes, time_bs_dpsgd)

    print("DPSGD accuracy by iterations:", results_dpsgd['iterations'])
    print("DPSGD runtime by iterations:", results_dpsgd['iterations_time'])
    print("RDPGD accuracy by iterations:", results_rdp['iterations'])
    print("RDPGD runtime by iterations:", results_rdp['iterations_time'])
    print("DPSGD accuracy by batch size:", results_dpsgd['batch_size'])
    print("DPSGD runtime by batch size:", results_dpsgd['batch_size_time'])

    plot_task4_results(results_dpsgd, results_rdp)
# %%

if __name__ == "__main__":
    task1()
    task2()
    #task3()
    #task4()

# %%
