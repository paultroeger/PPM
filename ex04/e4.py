#%% Imports and useful functions
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

#%%

# This is the gradient of the logistic loss
def gradient(weights, xi, yi):
    # yi = -1 if yi == 0 else 1
    exponent = yi * (xi.dot(weights))
    print(exponent)
    return - (yi*xi) / (1+np.exp(exponent))
    


def rdpgd(iterations, alpha, epsilon_bar):

    X_train, X_test, y_train, y_test = prepare_data(load_csv())

    weights = np.zeros(X_train.shape[1])
    n = X_train.shape[0]

    # everytime we add noise on the gradient we need privacy budget
    # using sequential composition we split epsilon
    epsilon_step = epsilon_bar / (iterations * n)
    sigma = np.sqrt(alpha / (2 * epsilon_step))

    for t in range(iterations):

        # gradient
        grad_sum = np.zeros_like(weights)
        for i in range(n):
            grad = gradient(weights, X_train[i], y_train[i])
        
            # do we clip here?
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad * (1.0 / grad_norm)

            # noise
            noise = np.random.normal(loc=0, scale=sigma, size=weights.shape)
            noisy_grad = grad + noise

            grad_sum += noisy_grad

        # weight update
        learning_rate = 0.1
        weights -= learning_rate * (grad_sum / n)

    model = LogisticRegression(max_iter=100)
    model.coef_ = weights.reshape(1, -1)
    model.intercept_ = np.array([0.0])
    
    model.classes_ = np.array([0, 1])

    return model.score(X_test, y_test)


def dpsgd(iterations, epsilon, delta, learning_rate, batch_size, C):
    
    X_train, X_test, y_train, y_test = prepare_data(load_csv())

    n = X_train.shape[0]
    weights = np.zeros(X_train.shape[1])

    # everytime we add noise on the gradient we need privacy budget
    # using sequential composition we split epsilon
    # worse case we touch each data point in each batch
    # that's why we need to split by batch_size * n
    epsilon_step = epsilon / (batch_size * n)
    sigma = C * np.sqrt(2 * np.log(1.25 / delta)) / epsilon_step

    for t in range(iterations):
        indices = np.random.choice(n, size=batch_size, replace=False)

        grad_sum = np.zeros_like(weights)
        for i in indices:
            g = gradient(weights, X_train[i], y_train[i])

            g_norm = np.linalg.norm(g)
            if g_norm > C:
                g = g * (C / g_norm)

            noise = np.random.normal(0, sigma, size=weights.shape)
            grad_sum += (g + noise)

        weights -= learning_rate * (grad_sum / batch_size)

    model = LogisticRegression(max_iter=100)
    model.coef_      = weights.reshape(1, -1)
    model.intercept_ = np.array([0.0])
    model.classes_   = np.array([0, 1])

    return model.score(X_test, y_test)

# Our functions
def load_csv():
    return pd.read_csv("./covid19-data-kaggle/covid.csv")

def prepare_data(df):
    df = df.drop(columns=["id"])
    df = df.replace([97,98,99], float('nan'))
    df = df.replace([2,3], [0,2])
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

#%% Tasks
def task1():
    print('-' * 6, 'Task 1', '-' * 6)
    df = load_csv()

    X_train, X_test, y_train, y_test = prepare_data(df)
   
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    print("Non-Private Accuracy:", model.score(X_test, y_test))

    print()

def task2():
    print('-' * 6, 'Task 2', '-' * 6)
    
    print("RDP Accuracy:", rdpgd(100, 2.0, 1.0))

    print()

def task3():
    print('-' * 6, 'Task 3', '-' * 6)

    print("DPS Accuracy:", dpsgd(100, 2.0, 1.0, 0.1, 10, 0.6))

    results = {}

    # default values
    it_def, eps_def, delta_def, lr_def, bs_def, c_def = 100, 1.0, 1e-5, 0.1, 32, 1.0

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
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5]
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

#%%

if __name__ == "__main__":
    task1()
    task2()
    #task3()

# %%
