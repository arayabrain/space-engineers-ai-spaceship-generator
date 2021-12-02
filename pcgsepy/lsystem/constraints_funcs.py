from typing import List
from .constraints import HLStructure
from .actions import AtomAction

from ..common.vecs import Vec


# TODO: Have these read from file
req_tiles = [
    'cockpit',
    'corridor',
    'thruster'
]

tiles_dimensions = {
    'cockpit': Vec.v3i(15, 15, 5),
    'corridor': Vec.v3i(15, 15, 5),
    'thruster': Vec.v3i(15, 15, 5)
}


def components_constraint(axiom: str) -> bool:
    components_ok = True
    for c in req_tiles:
        components_ok &= c in axiom
    return components_ok


def intersection_constraint(axiom: str) -> bool:
    global atoms_alphabet
    hl_structure = HLStructure()
    position = Vec.v3i(0, 0, 0)
    position_history = []
    rotations = []
    i = 0
    while i < len(axiom):
        for a in atoms_alphabet.keys():
            if axiom.startswith(a, i):
                action, args = atoms_alphabet[a]['action'], atoms_alphabet[a]['args']
                if action == AtomAction.MOVE:
                    direction = args.value.as_array()
                    for rot in reversed(rotations):
                        direction = rot.dot(direction)
                    position = position.sum(Vec.from_np(direction))
                elif action == AtomAction.ROTATE:
                    rotations.append(rotation_matrices[args])
                elif action == AtomAction.PUSH:
                    position_history.append(position)
                elif action == AtomAction.POP:
                    position = position_history.pop(-1)
                    if len(rotations) > 1:
                        rotations.pop(-1)
                elif action == AtomAction.PLACE:
                    dims = tiles_dimensions[a].as_array()
                    for r in reversed(rotations):
                        dims = r.dot(dims)
                    p = build_polyhedron(position=position,
                                         dims=Vec.from_np(dims))
                    hl_structure.add_hl_poly(p)
                i += len(a)
                break
    intersections = hl_structure.test_intersections()
    if intersections:
        hl_structure.show()
    return not intersections
