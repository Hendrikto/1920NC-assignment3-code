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
