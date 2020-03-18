import subprocess
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
)


def run_negsel(jar_path, train_path, test_path, r):
    """
    Run negative selection algorithm with the given arguments.

    Args:
        jar_path   = [str] file path to the negsel2.jar file
        train_path = [str] file path to the .train file for training
        test_path  = [str] file path to the .test or .txt file for testing
        r          = [int] pattern match length, 0 < r <= n

    Returns [bytes]:
        Stdout of the negative selection Java program.
    """
    return subprocess.run((
            f'java -jar {jar_path}'
            ' -c'
            ' -l'
            ' -n 10'
            f' -r {r}'
            f' -self {train_path}'
        ).split(),
        stdin=open(test_path),
        stdout=subprocess.PIPE,
    ).stdout


def num_matches(negsel, self_path, non_self_path, r):
    """
    Get number of pattern matches for the self and non-self classes.

    Args:
        negsel        = [fn] function to run the negative selection algorithm
        self_path     = [str] file path to the file with self strings
        non_self_path = [str] file path to the file with non-self strings
        r             = [int] pattern match length, 0 < r <= n

    Returns:
        y_true  = [list] self/non-self label of each test string as bool
        y_score = [list] logarithm of number of pattern matches of test strings
    """
    self_matches = negsel(test_path=self_path, r=r).split()
    y_true = [0] * len(self_matches)
    y_score = [float(matches) for matches in self_matches]

    non_self_matches = negsel(test_path=non_self_path, r=r).split()
    y_true += [1] * len(non_self_matches)
    y_score += [float(matches) for matches in non_self_matches]

    return y_true, y_score


def exercise_1_1(negsel):
    """
    Plot figure for Exercise 1 Part 1.

    Args:
        negsel = [fn] function to run the negative selection algorithm
    """
    y_true, y_score = num_matches(
        negsel,
        Path('negative-selection/english.test'),
        Path('negative-selection/tagalog.test'),
        4,
    )

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.4f}')
    plt.plot((0, 1), (0, 1), 'r--')
    plt.legend(loc='lower right')
    plt.show()
    plt.close()


def compute_aucs(negsel, non_self_path):
    """
    Compute Area Under the ROC Curve (AUC) for each r in the range [1, 10].

    Args:
        negsel        = [fn] function to run the negative selection algorithm
        non_self_path = [str] file path to the file with non-self strings

    Returns [list]:
        AUC value for each pattern match length given the language.
    """
    areas = []
    for r in range(1, 11):
        y_true, y_score = num_matches(
            negsel,
            Path('negative-selection/english.test'),
            non_self_path,
            r,
        )
        areas.append(roc_auc_score(y_true, y_score))

    return areas


def plot_aucs(areas, languages):
    """
    Plot areas under the ROC curves for the given languages.

    Args:
        areas     = [list] list with AUC scores for each language
        languages = [list] languages to include in the figure legend
    """
    plt.title('Area Under the ROC Curve')
    plt.xlabel('r')
    plt.ylabel('AUC')
    plt.xlim((1, 10))
    plt.ylim((0, 1))

    for area, language in zip(areas, languages):
        plt.plot(range(1, 11), area, label=language)

    plt.legend(loc='lower right')
    plt.show()
    plt.close()


def exercise_1_2(negsel):
    """
    Plot figure for Exercise 1 Part 2.

    Args:
        negsel = [fn] function to run the negative selection algorithm
    """
    areas = compute_aucs(negsel, Path('negative-selection/tagalog.test'))
    plot_aucs([areas], ['tagalog'])


def exercise_1_3(negsel):
    """
    Plot figure for Exercise 1 Part 3.

    Args:
        negsel = [fn] function to run the negative selection algorithm
    """
    languages = ['xhosa', 'hiligaynon', 'plautdietsch', 'middle-english']
    non_self_dir = Path('negative-selection/lang')
    areas = []
    for language in languages:
        areas.append(compute_aucs(negsel, non_self_dir / f'{language}.txt'))

    plot_aucs(areas, languages)


if __name__ == '__main__':
    negsel = partial(
        run_negsel,
        Path('negative-selection/negsel2.jar'),
        Path('negative-selection/english.train'),
    )

    exercise_1_1(negsel)
    exercise_1_2(negsel)
    exercise_1_3(negsel)
