from enum import Enum
import numpy as np


class AtomAction(str, Enum):
    """
    Enumerator for actions defined for an axiom's atom.
    """
    MOVE = 'move'
    PLACE = 'place'
    POP = 'pop'
    PUSH = 'push'
    ROTATE = 'rotate'


class Rotations(str, Enum):
    """
    Enumerator for all possible rotations.
    Format: 'Axis cw/ccw OtherAxis'.
    """
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


rotations_from_str = {
    'XcwY': Rotations.XcwY,
    'XcwZ': Rotations.XcwZ,
    'YcwX': Rotations.YcwX,
    'YcwZ': Rotations.YcwZ,
    'ZcwX': Rotations.ZcwX,
    'ZcwY': Rotations.ZcwY,
    'XccwY': Rotations.XccwY,
    'XccwZ': Rotations.XccwZ,
    'YccwX': Rotations.YccwX,
    'YccwZ': Rotations.YccwZ,
    'ZccwX': Rotations.ZccwX,
    'ZccwY': Rotations.ZccwY
}


# Rotation matrices for each rotation as NumPy matrices.
rotation_matrices = {
    Rotations.XcwY: np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    Rotations.XccwY: np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    Rotations.XcwZ: np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    Rotations.XccwZ: np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    Rotations.YcwX: np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    Rotations.YccwX: np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    Rotations.YcwZ: np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    Rotations.YccwZ: np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    Rotations.ZcwX: np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    Rotations.ZccwX: np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    Rotations.ZcwY: np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    Rotations.ZccwY: np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
}
