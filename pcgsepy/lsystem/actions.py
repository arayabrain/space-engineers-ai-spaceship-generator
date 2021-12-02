from enum import Enum
import numpy as np

from ..common.vecs import Orientation

class AtomAction(Enum):
    MOVE = 'move'
    PLACE = 'place'
    POP = 'pop'
    PUSH = 'push'
    ROTATE = 'rotate'


class Rotations(Enum):
    XcwY = 'XcwY'
    XcwZ = 'XcwZ'
    YcwX = 'YcwX'
    YcwZ = 'YcwZ'
    ZcwX = 'ZcwX'
    ZcwY = 'ZcwY'
    XccwY = 'XccwY'
    XccwZ = 'XccwZ'
    YccwX = 'YccwX'
    YccwZ = 'YccwZ'
    ZccwX = 'ZccwX'
    ZccwY = 'ZccwY'


rotation_matrices = {
    # R_z with -sinTheta
    Rotations.XcwY: np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    # R_z
    Rotations.XccwY: np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    # R_y with -sinTheta
    Rotations.XcwZ: np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    # R_y
    Rotations.XccwZ: np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),

    # R_z with -sinTheta
    Rotations.YcwX: np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    # R_z
    Rotations.YccwX: np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    # R_x with -sinTheta
    Rotations.YcwZ: np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    # R_x
    Rotations.YccwZ: np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 1]]),

    # R_y with -sinTheta
    Rotations.ZcwX: np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    # R_y
    Rotations.ZccwX: np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    # R_x with -sinTheta
    Rotations.ZcwY: np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    # R_x
    Rotations.ZccwY: np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 1]])
}