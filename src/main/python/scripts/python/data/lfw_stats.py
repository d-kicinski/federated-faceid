from collections import Counter
from pathlib import Path

import numpy as np
from tabulate import tabulate


def calc_lfw_statistics(data_dir: Path, min_examples: int = 5):
    examples_per_identity_stats = Counter()
    for identity_dir in data_dir.iterdir():
        example_num = len(list(identity_dir.iterdir()))
        examples_per_identity_stats[identity_dir.name] = example_num

    data = np.array(list(examples_per_identity_stats.values()))

    table = [(min_ex, len(data[data >= min_ex]), sum(data[data >= min_ex])) for min_ex in [1, 2, 5, 10, 20, 50, 100]]
    headers = ["min examples per identity", "identity no", "total training examples"]

    t = tabulate(table, headers, tablefmt="github")
    print(t)


if __name__ == '__main__':
    lfw_dir = Path("data/lfw")
    calc_lfw_statistics(lfw_dir)
