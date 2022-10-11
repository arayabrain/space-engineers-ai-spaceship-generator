from enum import Enum
from functools import cached_property
from multiprocessing.sharedctypes import Value
from os import stat
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class Vec:
    def __init__(self,
                 x: Union[int, float],
                 y: Union[int, float],
                 z: Optional[Union[int, float]] = None):
        """Create a vector.

        Args:
            x (Union[int, float]): The X value.
            y (Union[int, float]): The Y value.
            z (Optional[Union[int, float]], optional): The Z value. Defaults to None.
        """
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __eq__(self,
               other: 'Vec') -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.as_tuple())
    
    @classmethod
    def v2i(cls,
            x: int,
            y: int):
        """Create a 2D vector of integers.

        Args:
            x (int): The X value.
            y (int): The Y value.

        Returns:
            Vec: The 2D int vector.
        """
        return cls(x, y)

    @classmethod
    def v2f(cls,
            x: float,
            y: float):
        """Create a 2D vector of floats.

        Args:
            x (float): The X value.
            y (float): The Y value.

        Returns:
            Vec: The 2D float vector.
        """
        return cls(x, y)

    @classmethod
    def v3i(cls,
            x: int,
            y: int,
            z: int):
        """Create a 3D vector of integers.

        Args:
            x (int): The X value.
            y (int): The Y value.
            z (int): The Z value.

        Returns:
            Vec: The 3D int vector.
        """
        return cls(int(x),
                   int(y),
                   int(z))

    @classmethod
    def v3f(cls,
            x: float,
            y: float,
            z: float):
        """Create a 3D vector of floats.

        Args:
            x (float): The X value.
            y (float): The Y value.
            z (float): The Z value.

        Returns:
            Vec: The 3D float vector.
        """
        return cls(x, y, z)

    @classmethod
    def from_json(cls,
                  j: Dict[str, Any]) -> "Vec":
        """Create a vector from JSON data.

        Args:
            j (Dict[str, Any]): The JSON data.

        Returns:
            Vec: The vector.
        """
        return cls(j["X"], j["Y"], j.get("Z", None))

    @classmethod
    def from_np(cls,
                arr: Union[npt.NDArray[np.int32], npt.NDArray[np.float32]]) -> "Vec":
        """Create a vector from the NumPy array.

        Args:
            arr (np.ndarray): The NumPy array.

        Returns:
            Vec: The vector.
        """
        return cls(arr[0], arr[1]) if arr.size == 2 else cls(arr[0], arr[1], arr[2])

    @classmethod
    def from_tuple(cls,
                   tup: Union[Tuple[int, int], Tuple[int, int, int]]) -> "Vec":
        """Create a vector from the tuple.

        Args:
            tup (Union[Tuple[int, int], Tuple[int, int, int]]): The tuple.

        Returns:
            Vec: The vector.
        """
        return cls(tup[0], tup[1]) if len(tup) == 2 else cls(tup[0], tup[1], tup[2])

    def as_dict(self) -> Dict[str, Any]:
        """Convert the vector to a dictionary.

        Returns:
            Dict[str, Any]: The vector as dictionary.
        """
        s = {"X": self.x, "Y": self.y}
        if self.z is not None:
            s["Z"] = self.z
        return s

    def as_array(self) -> Union[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """Convert the vector to a NumPy array.

        Returns:
            Union[npt.NDArray[np.int32], npt.NDArray[np.float32]]: The vector as NumPy array.
        """
        return np.asarray([self.x, self.y]) if self.z is None else np.asarray([self.x, self.y, self.z])

    def as_tuple(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        """Convert the vector to a tuple.

        Returns:
            Union[Tuple[int, int], Tuple[int, int, int]]: The vector as tuple.
        """
        return (self.x, self.y) if self.z is None else (self.x, self.y, self.z)

    def to_veci(self) -> 'Vec':
        """Convert the current vector to int type.

        Returns:
            Vec: The vector as vector of ints.
        """
        return Vec(x=np.rint(self.x).astype(np.int32),
                   y=np.rint(self.y).astype(np.int32),
                   z=np.rint(self.z).astype(np.int32) if self.z is not None else None)

    def round(self,
              n: int = 1) -> 'Vec':
        """Round the vector values to a given precision.

        Args:
            n (int, optional): The rounding precision. Defaults to 1.

        Returns:
            Vec: The rounded vector.
        """
        return Vec(x=np.round(self.x, n),
                   y=np.round(self.y, n),
                   z=np.round(self.z, n) if self.z is not None else None)
    
    def floor(self) -> "Vec":
        """Apply the floor function to the vector.

        Returns:
            Vec: The floored vector.
        """
        return Vec(x=np.floor(self.x),
                   y=np.floor(self.y),
                   z=np.floor(self.z) if self.z is not None else None)
    
    @cached_property
    def max(self) -> Union[int, float]:
        """Compute the largest dimension of the vector.

        Returns:
            Union[int, float]: The largest dimension.
        """
        return max(self.x, self.y) if self.z is None else max(self.x, self.y, self.z)

    @cached_property
    def min(self) -> Union[int, float]:
        """Compute the smallest dimension of the vector.

        Returns:
            Union[int, float]: The smallest dimension.
        """
        return min(self.x, self.y) if self.z is None else min(self.x, self.y, self.z)

    def abs(self) -> "Vec":
        """Compute the absolute value of the vector.

        Returns:
            Vec: The positive vector.
        """
        return Vec(x=abs(self.x),
                   y=abs(self.y),
                   z=abs(self.z) if self.z is not None else None)

    def normalize(self) -> "Vec":
        """Normalize the vector (sum of elements equals to 1).

        Returns:
            Vec: The normalized vector.
        """
        m = self.max()
        return Vec(x=self.x / m,
                   y=self.y / m,
                   z=self.z / m if self.z is not None else None)

    def invert(self) -> "Vec":
        """Compute the inverse value of the vector.

        Returns:
            Vec: The inverse vector.
        """
        return Vec(x=1 / self.x,
                   y=1 / self.y,
                   z=1 / self.z if self.z is not None else None)

    def bbox(self,
             ignore_zero=True) -> Union[float, int]:
        """Compute the bounding box area/volume of the vector.

        Args:
            ignore_zero (bool, optional): Whether to ignore 0 in the computation. Defaults to True.

        Returns:
            Union[float, int]: The bounding box area/volume of the vector.
        """
        if ignore_zero:
            return self.x * self.y * self.z
        else:
            if self.x == 0 and self.y == 0 and self.z == 0:
                return 0
            else:
                x = self.x if self.x != 0 else 1
                y = self.y if self.y != 0 else 1
                z = self.z if self.z != 0 else 1
                return x * y * z

    def add(self,
            v: Union[float, int]) -> "Vec":
        """Add a scalar to the vector.

        Args:
            v (Union[float, int]): The scalar.

        Returns:
            Vec: The new vector.
        """
        return Vec(x=self.x + v,
                   y=self.y + v,
                   z=self.z + v if self.z is not None else None)
    
    def sum(self,
            other: "Vec") -> "Vec":
        """Compute the sum with another Vec.

        Args:
            other (Vec): The other vector.

        Raises:
            TypeError: Raised if the two vectors have different dimensions.

        Returns:
            Vec: The resulting vector.
        """
        try:
            return Vec(x=self.x + other.x,
                       y=self.y + other.y,
                       z=self.z + other.z)
        except TypeError:
            raise TypeError(
                f'Trying to sum mixed-dimension vectors: {self} + {other}')

    def diff(self,
             other: "Vec") -> "Vec":
        """Compute the difference with another Vec.

        Args:
            other (Vec): The other vector.

        Raises:
            TypeError: Raised if the two vectors have different dimensions.

        Returns:
            Vec: The resulting vector.
        """
        try:
            return Vec(x=self.x - other.x,
                       y=self.y - other.y,
                       z=self.z - other.z)
        except TypeError:
            raise TypeError(
                f'Trying to subtract mixed-dimension vectors: {self} - {other}')

    def dot(self,
            other: "Vec") -> "Vec":
        """Compute the dot-product with another Vec.

        Args:
            other (Vec): The other vector.

        Raises:
            TypeError: Raised if the two vectors have different dimensions.

        Returns:
            Vec: The resulting vector.
        """
        try:
            return Vec(x=self.x * other.x,
                       y=self.y * other.y,
                       z=self.z * other.z)
        except TypeError:
            raise TypeError(
                f'Trying to multiply mixed-dimension vectors: {self} * {other}')

    def scale(self,
              v: float) -> "Vec":
        """Scale vector by a value.

        Args:
            v (float): The scale factor.

        Returns:
            Vec: The scaled vector.
        """
        return Vec(x=self.x * v,
                   y=self.y * v,
                   z=self.z * v if self.z is not None else None)

    def opposite(self) -> "Vec":
        """Invert the direction along all dimensions.

        Returns:
            Vec: The opposite vector.
        """
        return self.scale(v=-1)
    
    @property
    def is_zero(self) -> bool:
        return self.x == 0 and self.y == 0 and (self.z == 0 if self.z is not None else True)
    
    @staticmethod
    def max(v1, v2) -> "Vec":
        assert (v1.z is not None and v2.z is not None) or (v1.z is None and v2.z is None), f"Comparing mixed-dimension vectors: {v1}, {v2}"
        return Vec(x=max(v1.x, v2.x),
                   y=max(v1.y, v2.y),
                   z=max(v1.z, v2.z) if v1.z is not None and v2.z is not None else None)
    
    @staticmethod
    def min(v1, v2) -> "Vec":
        assert (v1.z is not None and v2.z is not None) or (v1.z is None and v2.z is None), f"Comparing mixed-dimension vectors: {v1}, {v2}"
        return Vec(x=min(v1.x, v2.x),
                   y=min(v1.y, v2.y),
                   z=min(v1.z, v2.z) if v1.z is not None and v2.z is not None else None)


class Orientation(Enum):
    """Enum of different orientations. Values are the same used in the Space Engineer's API."""
    UP = Vec.v3i(0, 1, 0)
    DOWN = Vec.v3i(0, -1, 0)
    RIGHT = Vec.v3i(1, 0, 0)
    LEFT = Vec.v3i(-1, 0, 0)
    FORWARD = Vec.v3i(0, 0, -1)
    BACKWARD = Vec.v3i(0, 0, 1)


orientation_from_str = {
    'U': Orientation.UP,
    'D': Orientation.DOWN,
    'R': Orientation.RIGHT,
    'L': Orientation.LEFT,
    'F': Orientation.FORWARD,
    'B': Orientation.BACKWARD
}


def orientation_from_vec(vec: Vec) -> Orientation:
    """Get the orientation given its Vec.

    Args:
        vec (Vec): The vector orientation.

    Raises:
        ValueError: Raised if the vector is not a valid orientation.

    Returns:
        Orientation: The corresponding orientation.
    """
    if vec.x == 0:
        if vec.y == 0:
            if vec.z == -1:
                return Orientation.FORWARD
            elif vec.z == 1:
                return Orientation.BACKWARD
            else:
                raise ValueError(
                    f'Vector {vec} is not a valid orientation vector.')
        elif vec.y == 1:
            return Orientation.UP
        elif vec.y == -1:
            return Orientation.DOWN
        else:
            raise ValueError(
                f'Vector {vec} is not a valid orientation vector.')
    elif vec.x == 1:
        return Orientation.RIGHT
    elif vec.x == -1:
        return Orientation.LEFT
    else:
        raise ValueError(f'Vector {vec} is not a valid orientation vector.')


character_camera_dist = Vec.v3f(0., 1.6369286, 0.)


def get_rotation_matrix(forward: Vec,
                        up: Vec) -> npt.NDArray[np.float32]:
    """Compute the rotation matrix from the forward and up vectors.

    Args:
        forward (Vec): The forward vector.
        up (Vec): The up vector.

    Returns:
        npt.NDArray[np.float32]: The rotation matrix.
    """
    f = forward.as_array()
    u = up.as_array()
    z = f / np.sqrt(np.dot(f, f))
    y = u / np.sqrt(np.dot(u, u))
    x = np.cross(z, y)
    return np.column_stack((x, y, -z))


def rotate(rotation_matrix: npt.NDArray[np.float32],
           vector: Vec) -> Vec:
    """Rotate a vector using a rotation matrix.

    Args:
        rotation_matrix (npt.NDArray[np.float32]): The rotation matrix.
        vector (Vec): The vector.

    Returns:
        Vec: The rotated vector.
    """
    v = vector.as_array()
    v = np.dot(rotation_matrix, v)
    return Vec.from_np(v)
