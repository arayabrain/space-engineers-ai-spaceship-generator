import numpy as np
from enum import Enum
from typing import Any, Dict, Dict, Tuple

def vec2(x: int,
         y: int) -> Dict[str, int]:
    return {
        "X": x,
        "Y": y
    }

def vec3(x: int,
         y: int,
         z: int) -> Dict[str, int]:
    return {
        "X": x,
        "Y": y,
        "Z": z
    }

def vec2f(x: float,
         y: float) -> Dict[str, float]:
    return {
        "X": x,
        "Y": y
    }

def vec3f(x: float,
         y: float,
         z: float) -> Dict[str, float]:
    return {
        "X": x,
        "Y": y,
        "Z": z
    }

def sum_vecs(a: Dict[str, Any],
             b: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "X": a["X"] + b["X"],
        "Y": a["Y"] + b["Y"],
        "Z": a["Z"] + b["Z"],
    }

def sum_tuples(a: Tuple[Any],
               b: Tuple[Any]) -> Tuple[Any]:
    assert len(a) == len(b), f'Can\'t sum tuples of different size ({len(a)} and {len(b)})'
    r = []
    for i in range(len(a)):
        r.append(type(a[i])(a[i] + b[i]))
    return tuple(r)

def to_list(a: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (a["X"], a["Y"], a["Z"])

character_camera_dist = vec3f(0., 1.6369286, 0.)

def get_rotation_matrix(forward: Dict[str, Any],
                        up: Dict[str, Any]) -> np.ndarray:
    f = np.asarray(to_list(forward))
    u = np.asarray(to_list(up))
    z = f / np.sqrt(np.dot(f, f))
    y = u / np.sqrt(np.dot(u, u))
    x = np.cross(z, y)
    return np.column_stack((x, y, -z))

def rotate(rotation_matrix: np.ndarray,
           vector: Dict[str, Any]) -> Dict[str, Any]:
    v = np.asarray(to_list(vector))
    v = np.dot(rotation_matrix, v)
    return vec3f(v[0], v[1], v[2])


class Orientation(Enum):
    UP = vec3f(0., 1., 0.)
    DOWN = vec3f(0., -1., 0.)
    RIGHT = vec3f(1., 0., 0.)
    LEFT = vec3f(-1., 0., 0.)
    FORWARD = vec3f(0., 0., -1.)
    BACKWARD = vec3f(0., 0., 1.)
