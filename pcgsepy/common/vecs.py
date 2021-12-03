import numpy as np
from enum import Enum
from typing import Any, Dict, Dict, Tuple


class Vec:
    def __init__(self,
                 x: Any,
                 y: Any,
                 z: Any = None):
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def v2i(cls,
            x: int,
            y: int):
        return cls(x, y)
    
    @classmethod
    def v2f(cls,
            x: float,
            y: float):
        return cls(x, y)
    
    @classmethod
    def v3i(cls,
            x: int,
            y: int,
            z: int):
        return cls(x, y, z)
    
    @classmethod
    def v3f(cls,
            x: float,
            y: float,
            z: float):
        return cls(x, y, z)
    
    @classmethod
    def from_json(cls,
                  j: Dict[str, Any]):
        if "Z" in j.keys():
            return cls(j["X"], j["Y"], j["Z"])
        else:
            return cls(j["X"], j["Y"])
    
    @classmethod
    def from_np(cls, arr):
        if arr.size == 2:
            return cls(arr[0], arr[1])
        else:
            return cls(arr[0], arr[1], arr[2])

    def as_dict(self) -> Dict[str, Any]:
        s = {"X": self.x, "Y": self.y}
        if self.z is not None:
            s["Z"] = self.z
        return s
    
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
    
    def largest_dim(self) -> Any:
        if self.x >= self.y and (self.z is None or self.x >= self.z):
            return self.x
        elif self.y >= self.x and (self.z is None or self.y >= self.z):
            return self.y
        else:
            return self.z

    def smallest_dim(self) -> Any:
        if self.x <= self.y and (self.z is None or self.x <= self.z):
            return self.x
        elif self.y <= self.x and (self.z is None or self.y <= self.z):
            return self.y
        else:
            return self.z
    
    def sum(self,
            other: "Vec") -> "Vec":
        if self.z is not None and other.z is not None:
            return Vec(x=self.x + other.x,
                       y=self.y + other.y,
                       z=self.z + other.z)
        elif self.z is None and other.z is None:
            return Vec(x=self.x + other.x,
                       y=self.y + other.y)
        else:
            raise Exception(f'Trying to sum mixed-dimension vectors: {self} + {other}')
    
    def as_array(self) -> np.ndarray:
        if self.z is not None:
            return np.asarray([self.x, self.y, self.z])
        else:
            return np.asarray([self.x, self.y])
    
    def as_tuple(self):
        if self.z is not None:
            return (self.x, self.y, self.z)
        else:
            return (self.x, self.y)

class Orientation(Enum):
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

def orientation_from_vec(vec) -> Orientation:
    if vec.x == 0:
        if vec.y == 0:
            if vec.z == 1:
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
