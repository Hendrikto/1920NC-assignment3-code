import subprocess

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
)


def run_negsel(jar_path, alphabet_path, train_path, test_path, n, r):
    return subprocess.run((
            f'java -jar {jar_path}'
            f' -alphabet file://{alphabet_path}'
            f' -self {train_path}'
            f' -n {n}'
            f' -r {r}'
            ' -c'
        ).split(),
        stdin=open(test_path, 'r'),
        stdout=subprocess.PIPE,
    ).stdout


def aggregate_scores(output):
    scores = []
    for line in output.splitlines():
        numbers = line.split()
        length = len(numbers)
        numbers = map(float, numbers)
        numbers = map(lambda x: x / length, numbers)
        scores.append(sum(numbers))
    return scores


def plot_roc(y_true, y_pred, n, r):
    roc = roc_curve(y_true, y_pred)

    plt.figure()
    plt.title(f'Receiver Operating Characteristic (${n = }, {r = }$)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot((0, 1), (0, 1), linestyle='--')
    plt.plot(roc[0], roc[1], label=f'ROC curve (AUC = {auc(roc[0], roc[1]):.4f})')
    plt.legend()
    plt.show()
    plt.close()


def calculate_roc_auc_scores(jar_path, alphabet_path, train_path, test_path, y_true, n_max):
    auc_scores = np.full((n_max,) * 2, np.nan)
    for n in range(1, n_max + 1):
        for r in range(1, n + 1):
            output = run_negsel(jar_path, alphabet_path, train_path, test_path, n, r)
            scores = aggregate_scores(output)
            auc_score = roc_auc_score(y_true, scores)
            auc_scores[n - 1, r - 1] = auc_score
            print(f'ROC AUC score ({n=}, {r=}): {auc_score:.4f}')
    return auc_scores
