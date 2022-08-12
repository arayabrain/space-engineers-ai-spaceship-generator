from typing import Any, Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt


class EmptyBufferException(Exception):
    pass


def mean_merge(x1: Any,
               x2: Any) -> Any:
    """Return the average of two values.

    Args:
        x1 (Any): The first value.
        x2 (Any): The second value.

    Returns:
        Any: The average of the two values.
    """
    return (x1 + x2) / 2


def max_merge(x1: Any,
              x2: Any) -> Any:
    """Return the maximum of two values.

    Args:
        x1 (Any): The first value.
        x2 (Any): The second value.

    Returns:
        Any: The maximum of the two values.
    """
    return max(x1, x2)


def min_merge(x1: Any,
              x2: Any) -> Any:
    """Return the minimum of two values.

    Args:
        x1 (Any): The first value.
        x2 (Any): The second value.

    Returns:
        Any: The minimum of the two values.
    """
    return min(x1, x2)


merge_methods = {
    'mean_merge': mean_merge,
    'max_merge': max_merge,
    'min_merge': min_merge
}


class Buffer:
    def __init__(self,
                 merge_method: Callable[[Any, Any], Any] = mean_merge) -> None:
        """Create an empty buffer.

        Args:
            merge_method (Callable[[Any, Any], Any], optional): The merging method. Defaults to `mean_merge`.
        """
        self._xs, self._ys = [], []
        self._merge = merge_method

    def _contains(self,
                  x: Any) -> int:
        """Check whether the element is present in the buffer. If it is, the index of the element is returned, otherwise `-1` is returned.

        Args:
            x (Any): The element to check.

        Returns:
            int: The index of the element if it is present, otherwise `-1`.
        """
        for i, _x in enumerate(self._xs):
            if np.array_equal(x, _x):
                return i
        return -1

    def insert(self,
               x: Any,
               y: npt.NDArray[np.float32]) -> None:
        """Add a datapoint to the buffer.

        Args:
            x (Any): The input data.
            y (Any): The label data.
        """
        i = self._contains(x)
        if i > -1:
            y0 = self._ys[i]
            self._ys[i] = self._merge(y0, y)
        else:
            self._xs.append(np.asarray(x))
            self._ys.append(y)

    def get(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Get the array representation of the buffer.

        Raises:
            EmptyBufferException: Raised if the buffer is empty.

        Returns:
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: The input data and label data as NumPy arrays.
        """
        if len(self._xs) > 0:
            xs = np.empty((len(self._xs), len(self._xs[0])))
            for i, X in enumerate(self._xs):
                xs[i, :] = X
            ys = np.asarray(self._ys)
            return xs, ys
        else:
            raise EmptyBufferException('Buffer is empty!')

    def clear(self) -> None:
        """Clear the buffer."""
        self._xs, self._ys = [], []

    def to_json(self) -> Dict[str, Any]:
        return {
            'xs': [x.tolist() for x in self._xs],
            'ys': [y.tolist() for y in self._ys],
            'merge_method': self._merge.__name__
        }

    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'Buffer':
        b = Buffer(merge_method=merge_methods[my_args['merge_method']])
        b._xs = [np.asarray(x) for x in my_args['xs']]
        b._ys = [np.asarray(y) for y in my_args['ys']]
        return b
