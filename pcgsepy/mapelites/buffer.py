from typing import Any, Tuple

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
                 merge_method: callable[[Any, Any], Any] = mean_merge) -> None:
        self._xs = []
        self._ys = []
        self._merge = merge_method
        
    def contains(self,
                 x: Any) -> bool:
        return x in self._xs
    
    def insert(self,
               x: Any,
               y: Any) -> None:
        if self.contains(x):
            i = self._xs.index(x)
            y0 = self._ys[i]
            self._ys[i] = self._merge(y0, y)
        else:
            self._xs.append(x)
            self._ys.append(y)
    
    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._xs) > 0:
            return np.asarray(self._xs), np.asarray(self._ys)
        else:
            raise EmptyBufferException('Buffer is empty!')