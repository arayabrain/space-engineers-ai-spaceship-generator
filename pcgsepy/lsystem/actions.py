from enum import Enum, auto
import numpy as np

from ..common.vecs import Orientation

class AtomAction(Enum):
    PLACE = auto()
    MOVE = auto()
    PUSH = auto()
    ROTATE = auto()
    POP = auto()


class Rotations(Enum):
    XcwY = auto()
    XcwZ = auto()
    YcwX = auto()
    YcwZ = auto()
    ZcwX = auto()
    ZcwY = auto()
    XccwY = auto()
    XccwZ = auto()
    YccwX = auto()
    YccwZ = auto()
    ZccwX = auto()
    ZccwY = auto()


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

atoms_alphabet = {
    "+": {"action": AtomAction.MOVE, "args": Orientation.RIGHT},
    "-": {"action": AtomAction.MOVE, "args": Orientation.LEFT},
    "!": {"action": AtomAction.MOVE, "args": Orientation.UP},
    "?": {"action": AtomAction.MOVE, "args": Orientation.DOWN},
    ">": {"action": AtomAction.MOVE, "args": Orientation.FORWARD},
    "<": {"action": AtomAction.MOVE, "args": Orientation.BACKWARD},
    "RotXcwY": {"action": AtomAction.ROTATE, "args": Rotations.XcwY},
    "RotXcwZ": {"action": AtomAction.ROTATE, "args": Rotations.XcwZ},
    "RotYcwX": {"action": AtomAction.ROTATE, "args": Rotations.YcwX},
    "RotYcwZ": {"action": AtomAction.ROTATE, "args": Rotations.YcwZ},
    "RotZcwX": {"action": AtomAction.ROTATE, "args": Rotations.ZcwX},
    "RotZcwY": {"action": AtomAction.ROTATE, "args": Rotations.ZcwY},
    "RotXccwY": {"action": AtomAction.ROTATE, "args": Rotations.XccwY},
    "RotXccwZ": {"action": AtomAction.ROTATE, "args": Rotations.XccwZ},
    "RotYccwX": {"action": AtomAction.ROTATE, "args": Rotations.YccwX},
    "RotYccwZ": {"action": AtomAction.ROTATE, "args": Rotations.YccwZ},
    "RotZccwX": {"action": AtomAction.ROTATE, "args": Rotations.ZccwX},
    "RotZccwY": {"action": AtomAction.ROTATE, "args": Rotations.ZccwY},
    "[": {"action": AtomAction.PUSH, "args": []},
    "]": {"action": AtomAction.POP, "args": []},
    "cockpit": {"action": AtomAction.PLACE, "args": []},
    "corridor": {"action": AtomAction.PLACE, "args": []},
    "thruster": {"action": AtomAction.PLACE, "args": []},
}