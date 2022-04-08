from typing import Any, Callable, Tuple

import numpy as np

class EmptyBufferException(Exception):
    pass

def mean_merge(x1, x2):
    return (x1 + x2) / 2

def max_merge(x1, x2):
    return max(x1, x2)

def min_merge(x1, x2):
    return min(x1, x2)

class Buffer:
    def __init__(self,
                 merge_method: Callable[[Any, Any], Any] = mean_merge) -> None:
        self._xs = []
        self._ys = []
        self._merge = merge_method
        
    def contains(self,
                 x: Any) -> int:
        for i, _x in enumerate(self._xs):
            if np.array_equal(x, _x):
                return i
        return -1
    
    def insert(self,
               x: Any,
               y: Any) -> None:
        i = self.contains(x)
        if i > -1:
            y0 = self._ys[i]
            self._ys[i] = self._merge(y0, y)
        else:
            self._xs.append(np.asarray(x))
            self._ys.append(y)
    
    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._xs) > 0:
            xs = np.empty((len(self._xs), len(self._xs[0])))
            for i, X in enumerate(self._xs):
                xs[i, :] = X
            ys = np.asarray(self._ys)
            return xs, ys
        else:
            raise EmptyBufferException('Buffer is empty!')