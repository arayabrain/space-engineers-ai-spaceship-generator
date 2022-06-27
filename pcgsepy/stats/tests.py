import itertools
from typing import List, Tuple

from numpy.typing import NDArray
from scipy.stats import f_oneway, kruskal, shapiro

THRESHOLD_PVALUE = 0.05


def shapiro_wilk(samples: List[NDArray]) -> List[Tuple[float, float]]:
    stats = []
    for sample in samples:
        shapiro_test = shapiro(sample)
        stats.append((shapiro_test.statistic, shapiro_test.pvalue))
    return stats

def anova(samples: List[NDArray]) -> List[Tuple[float, float]]:
    stats = []
    stats.append(f_oneway(*samples))
    if stats[0][1] < THRESHOLD_PVALUE:
        for comb in itertools.combinations(iterable=samples,
                                           r=2):
            stats.append(f_oneway(*comb))
    return stats

def kruskal_wallis(samples: List[NDArray]) -> List[Tuple[float, float]]:
    stats = []
    kruskal_test = kruskal(*samples)
    stats.append((kruskal_test.statistic, kruskal_test.pvalue))
    if stats[0][1] < THRESHOLD_PVALUE:
        for comb in itertools.combinations(iterable=samples,
                                           r=2):
            kruskal_test = kruskal(*comb)
            stats.append((kruskal_test.statistic, kruskal_test.pvalue))
    return stats
