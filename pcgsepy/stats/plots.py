from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_rankings(samples: List[NDArray],
                  labels: List[str],
                  names: List[str],
                  title: str,
                  filename: Optional[str]):
    
    def __count_score(arr: NDArray,
                      v: int) -> int:
        return np.sum([1 if x == v else 0 for x in arr])

    data = {}
    for i, label in enumerate(labels):
        data[label] = []
        for sample in samples:
            data[label].append(__count_score(arr=sample,
                                             v=i + 1))
    df = pd.DataFrame(data=data,
                      index=names)
    ax = df.plot.barh()
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_scores(samples: List[NDArray],
                names: List[str],
                score_to_value: Dict[int, float],
                title: str,
                filename: Optional[str]):
    all_values = []

    for sample in samples:
        all_values.append([])
        for e in sample:
            all_values[-1].append(score_to_value[e])

    plt.bar(names, height=[np.sum(x) for x in all_values])
    plt.title(title)
    # plt.xticks(rotation = 45)
    if filename:
        plt.savefig(filename)
    plt.show()
