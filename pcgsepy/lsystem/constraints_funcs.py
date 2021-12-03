import numpy as np
from typing import Any, Dict, List, Tuple
from .constraints import HLStructure
from .actions import AtomAction
from .structure_maker import StructureMaker
from ..common.vecs import Vec, Orientation
from ..structure import Structure


def components_constraint(axiom: str,
                          extra_args: Dict[str, Any]) -> bool:
    req_tiles = extra_args['req_tiles']
    
    components_ok = True
    for c in req_tiles:
        components_ok &= c in axiom
    return components_ok


def intersection_constraint(axiom: str,
                            extra_args: Dict[str, Any]) -> bool:
    position = Vec.v3i(0, 0, 0)
    hl_structure = StructureMaker(atoms_alphabet=extra_args['alphabet'],
                                  position=position).fill_structure(structure=None,
                                                                    axiom=axiom,
                                                                    additional_args={
                                                                        'tiles_dimensions': extra_args['tiles_dimensions']
                                                                    })
    intersections = hl_structure.test_intersections()
    if intersections:
        hl_structure.show()
    return not intersections

def symmetry_constraint(axiom: str,
                        extra_args: Dict[str, Any]) -> bool:
    # get low-level structure representation
    base_position, orientation_forward, orientation_up = Vec.v3i(0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = StructureMaker(atoms_alphabet=extra_args['alphabet'],
                               position=base_position).fill_structure(structure=structure,
                                                                      axiom=axiom,
                                                                      additional_args={}).as_array()
    is_symmetric = False
    for dim in range(3):
        is_symmetric |= np.array_equal(structure, np.flip(structure, axis=dim))
    return is_symmetric

def wheels_plane_constraint(axiom: str,
                            extra_args: Dict[str, Any]) -> bool:
    # get low-level blocks (cockpit and wheels)
    base_position, orientation_forward, orientation_up = Vec.v3i(0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    blocks = StructureMaker(atoms_alphabet=extra_args['alphabet'],
                            position=base_position).fill_structure(structure=structure,
                                                                   axiom=axiom,
                                                                   additional_args={}).get_all_blocks()
    cockpit = None
    wheels = []
    for b in blocks:
        if b.block_type == 'LargeBlockCockpit':
            cockpit = b
        elif b.block_type == 'OffroadSmallRealWheel1x1':
            wheels.append(b)
    # if there are no wheels, the constraint is unsat
    if len(wheels) == 0:
        return False
    else:
        # create groups of wheels on the same plane
        wheel_groups = []
        for wheel in wheels:
            added = False
            for group in wheel_groups:
                ref_wheel = group["wheels"][0]
                ref_x, ref_y, ref_z = ref_wheel.position.as_tuple()
                x, y, z = wheel.position.as_tuple()
                if group["on"] == "any":
                    if x == ref_x:
                        group["on"] = "x"
                        group["wheels"].append(wheel)
                        added = True
                        break
                    elif y == ref_y:
                        group["on"] = "y"
                        group["wheels"].append(wheel)
                        added = True
                        break
                    elif z == ref_z:
                        group["on"] = "z"
                        group["wheels"].append(wheel)
                        added = True
                        break
                elif group["on"] == "x":
                    if x == ref_x:
                        group["wheels"].append(wheel)
                        added = True
                        break
                elif group["on"] == "y":
                    if y == ref_y:
                        group["wheels"].append(wheel)
                        added = True
                        break
                elif group["on"] == "z":
                    if z == ref_z:
                        group["wheels"].append(wheel)
                        added = True
                        break
            if not added:
                wheel_groups.append({"on": "any", "wheels": [wheel]})
        # check that at least one group of wheels is on same orientation UP as the cockpit
        sat = False
        for group in wheel_groups:
            sat |= group['wheels'][0].orientation_up == cockpit.orientation_up
        return sat
