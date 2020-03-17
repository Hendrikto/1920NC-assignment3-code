import subprocess
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

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


def calculate_roc_auc_scores(negsel, y_true, n_max):
    auc_scores = np.full((n_max,) * 2, np.nan)
    for n in range(1, n_max + 1):
        for r in range(1, n + 1):
            output = negsel(n=n, r=r)
            scores = aggregate_scores(output)
            auc_score = roc_auc_score(y_true, scores)
            auc_scores[n - 1, r - 1] = auc_score
            print(f'ROC AUC score ({n=}, {r=}): {auc_score:.4f}')
    return auc_scores


def plot_roc_auc_scores(scores):
    mean_score = np.nanmean(scores)
    n_max = scores.shape[0]

    plt.figure()
    plt.title('ROC AUC Scores')
    plt.xlabel('$r$')
    plt.ylabel('$n$')
    plt.xticks(range(n_max), range(1, n_max + 1))
    plt.yticks(range(n_max), range(1, n_max + 1))
    plt.imshow(scores)
    for n in range(n_max):
        for r in range(n + 1):
            auc_score = scores[n, r]
            plt.text(
                r, n,
                f'{auc_score:.4f}',
                color='white' if auc_score < mean_score else 'black',
                ha='center',
                va='center',
            )
    plt.colorbar()
    plt.show()
    plt.close()


parser = ArgumentParser()
parser.add_argument(
    '--alphabet',
    type=Path,
    default='snd-cert.alpha',
    dest='alphabet_file',
)
parser.add_argument(
    '--data_dir',
    type=Path,
    default='negative-selection/syscalls/snd-cert',
)
parser.add_argument(
    '--jar',
    type=Path,
    default='negative-selection/negsel2.jar',
    dest='jar_path',
)
parser.add_argument(
    '--label',
    type=Path,
    default='snd-cert.1.labels',
    dest='label_file',
)
parser.add_argument(
    '--test',
    type=Path,
    default='snd-cert.1.test',
    dest='test_file',
)
parser.add_argument(
    '--train',
    type=Path,
    default='snd-cert.train',
    dest='train_file',
)


if __name__ == '__main__':
    args = parser.parse_args()

    y_true = np.loadtxt(args.data_dir / args.label_file, dtype=np.bool)

    with open(args.data_dir / args.train_file) as train_file:
        # smallest line length, excluding '\n'
        n_max = min(map(len, train_file)) - 1
        print('Inferred max n:', n_max)

    negsel = partial(
        run_negsel,
        args.jar_path,
        args.data_dir / args.alphabet_file,
        args.data_dir / args.train_file,
        args.data_dir / args.test_file,
    )
    roc_auc_scores = calculate_roc_auc_scores(negsel, y_true, n_max)
    plot_roc_auc_scores(roc_auc_scores)

    n, r = np.unravel_index(np.nanargmax(roc_auc_scores), roc_auc_scores.shape)
    n += 1
    r += 1
    plot_roc(y_true, aggregate_scores(negsel(n, r)), n, r)
