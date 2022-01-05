import numpy as np
from enum import Enum
from typing import Any, Dict, Union


class Vec:
    """
    Generic vector class.
    """

    def __init__(self,
                 x: Any,
                 y: Any,
                 z: Any = None):
        """
        Create a vector.

        Parameters
        ----------
        x : Any
            The X value
        y : Any
            The Y value
        z : Any
            The Z value
        """
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return str(self.as_dict())

    def __repr__(self) -> str:
        return str(self.as_dict())

    def __eq__(self,
               other: 'Vec') -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    @classmethod
    def v2i(cls,
            x: int,
            y: int):
        """
        Create a 2D vector of integers.

        Parameters
        ----------
        x : int
            The X value
        y : int
            The Y value

        Returns
        -------
        Vec
            The vector.
        """
        return cls(x, y)

    @classmethod
    def v2f(cls,
            x: float,
            y: float):
        """
        Create a 2D vector of floats.

        Parameters
        ----------
        x : float
            The X value
        y : float
            The Y value

        Returns
        -------
        Vec
            The vector.
        """
        return cls(x, y)

    @classmethod
    def v3i(cls,
            x: int,
            y: int,
            z: int):
        """
        Create a 3D vector of integers.

        Parameters
        ----------
        x : int
            The X value
        y : int
            The Y value
        z : int
            The Z value

        Returns
        -------
        Vec
            The vector.
        """
        return cls(x, y, z)

    @classmethod
    def v3f(cls,
            x: float,
            y: float,
            z: float):
        """
        Create a 3D vector of floats.

        Parameters
        ----------
        x : float
            The X value
        y : float
            The Y value
        z : float
            The Z value

        Returns
        -------
        Vec
            The vector.
        """
        return cls(x, y, z)

    @classmethod
    def from_json(cls,
                  j: Dict[str, Any]):
        """
        Create a vector from JSON data.

        Parameters
        ----------
        j : Dict[str, Any]
            The JSON data.

        Returns
        -------
        Vec
            The vector.
        """
        if "Z" in j.keys():
            return cls(j["X"], j["Y"], j["Z"])
        else:
            return cls(j["X"], j["Y"])

    @classmethod
    def from_np(cls,
                arr: np.ndarray):
        """
        Create a vector from the NumPy array.

        Parameters
        ----------
        arr : np.ndarray
            The NumPy array.

        Returns
        -------
        Vec
            The vector.
        """
        if arr.size == 2:
            return cls(arr[0], arr[1])
        else:
            return cls(arr[0], arr[1], arr[2])

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the vector to a dictionary.

        Returns
        -------
        Dict[str, Any]
            The vector as a dictionary.
        """
        s = {"X": self.x, "Y": self.y}
        if self.z is not None:
            s["Z"] = self.z
        return s

    def as_array(self) -> np.ndarray:
        """
        Convert the vector to a NumPy array.

        Returns
        -------
        np.ndarray
            The NumPy array.
        """
        if self.z is not None:
            return np.asarray([self.x, self.y, self.z])
        else:
            return np.asarray([self.x, self.y])

    def as_tuple(self):
        """
        Convert the vector to a tuple of values.

        Returns
        -------
        Tuple[Union[int, float]]
            A 2- or 3-dimensional tuple.
        """
        if self.z is not None:
            return (self.x, self.y, self.z)
        else:
            return (self.x, self.y)

    def largest_dim(self) -> Union[int, float]:
        """
        Compute the largest dimension of the vector.

        Returns
        -------
        Union[int, float]
            The largest dimension.
        """
        if self.x >= self.y and (self.z is None or self.x >= self.z):
            return self.x
        elif self.y >= self.x and (self.z is None or self.y >= self.z):
            return self.y
        else:
            return self.z

    def smallest_dim(self) -> Any:
        """
        Compute the smallest dimension of the vector.

        Returns
        -------
        Union[int, float]
            The smallest dimension.
        """
        if self.x <= self.y and (self.z is None or self.x <= self.z):
            return self.x
        elif self.y <= self.x and (self.z is None or self.y <= self.z):
            return self.y
        else:
            return self.z

    def sum(self,
            other: "Vec") -> "Vec":
        """
        Compute the sum with another Vec.

        Parameters
        ----------
        other : Vec
            The other Vec.

        Returns
        -------
        Vec
            The resulting vector.
        """
        if self.z is not None and other.z is not None:
            return Vec(x=self.x + other.x,
                       y=self.y + other.y,
                       z=self.z + other.z)
        elif self.z is None and other.z is None:
            return Vec(x=self.x + other.x,
                       y=self.y + other.y)
        else:
            raise Exception(f'Trying to sum mixed-dimension vectors: {self} + {other}')


class Orientation(Enum):
    """
    Enum of different orientations.
    Values are the same used in the Space Engineer's API.
    """
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
    """
    Get the orientation given its Vec.

    Parameters
        ----------
        vec : Vec
            The Vec.

        Returns
        -------
        Orientation
            The corresponding Orientation.
    """
    if vec.x == 0:
        if vec.y == 0:
            if vec.z == -1:
                return Orientation.FORWARD
            else:
                return Orientation.BACKWARD
        elif vec.y == 1:
            return Orientation.UP
        else:
            return Orientation.DOWN
    elif vec.x == 1:
        return Orientation.RIGHT
    else:
        return Orientation.LEFT


character_camera_dist = Vec.v3f(0., 1.6369286, 0.)

# def get_rotation_matrix(forward: Vec,
#                         up: Vec) -> np.ndarray:
#     f = forward.as_array()
#     u = up.as_array()
#     z = f / np.sqrt(np.dot(f, f))
#     y = u / np.sqrt(np.dot(u, u))
#     x = np.cross(z, y)
#     return np.column_stack((x, y, -z))

# def rotate(rotation_matrix: np.ndarray,
#            vector: Vec) -> Vec:
#     v = vector.as_array()
#     v = np.dot(rotation_matrix, v)
#     return Vec.from_np(v)