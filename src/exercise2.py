import subprocess


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
