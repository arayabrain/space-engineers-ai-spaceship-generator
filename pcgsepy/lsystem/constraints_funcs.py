import numpy as np
from typing import Any, Dict

from ..structure import Structure, IntersectionException
from ..common.vecs import Vec, Orientation
from .structure_maker import LLStructureMaker
from ..config import MAME_MEAN, MAME_STD, MAMI_MEAN, MAMI_STD


def components_constraint(axiom: str, extra_args: Dict[str, Any]) -> bool:
    req_tiles = extra_args['req_tiles']

    components_ok = True
    for c in req_tiles:
        components_ok &= c in axiom
    return components_ok


def intersection_constraint(axiom: str, extra_args: Dict[str, Any]) -> bool:
    base_position, orientation_forward, orientation_up = Vec.v3i(
        0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    try:
        structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                     position=base_position).fill_structure(
                                         structure=structure,
                                         axiom=axiom,
                                         additional_args={
                                             'intersection_checking': True
                                         })
        structure.update(origin=base_position,
                         orientation_forward=orientation_forward,
                         orientation_up=orientation_up)
        # check block intersecting after placement
        max_x, max_y, max_z = structure._max_dims
        matrix = np.zeros(shape=(max_x + 5, max_y + 5, max_z + 5),
                          dtype=np.uint8)
        for i, j, k in structure._blocks.keys():
            block = structure._blocks[(i, j, k)]
            r = block.size
            if np.sum(matrix[i:i + r, j:j + r, k:k + r]) == 0:
                matrix[i:i + r, j:j + r, k:k + r] = 1
            else:
                return False
    except IntersectionException:
        return False
    return True


def symmetry_constraint(axiom: str, extra_args: Dict[str, Any]) -> bool:
    # get low-level structure representation
    base_position, orientation_forward, orientation_up = Vec.v3i(
        0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                 position=base_position).fill_structure(
                                     structure=structure,
                                     axiom=axiom,
                                     additional_args={})
    structure.update(origin=base_position,
                     orientation_forward=orientation_forward,
                     orientation_up=orientation_up)

    structure = structure.as_array()

    is_symmetric = False
    for dim in range(3):
        is_symmetric |= np.array_equal(structure, np.flip(structure, axis=dim))
    return is_symmetric


def axis_constraint(axiom: str, extra_args: Dict[str, Any]) -> bool:
    # get low-level structure representation
    base_position, orientation_forward, orientation_up = Vec.v3i(
        0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                 position=base_position).fill_structure(
                                     structure=structure,
                                     axiom=axiom,
                                     additional_args={})
    structure.update(origin=base_position,
                     orientation_forward=orientation_forward,
                     orientation_up=orientation_up)

    volume = structure.as_array().shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(list(volume)))
    mame = largest_axis / medium_axis
    mami = largest_axis / smalles_axis    
    sat = True
    sat &= MAME_MEAN - MAME_STD <= mame <= MAME_MEAN + MAME_STD
    sat &= MAMI_MEAN - MAMI_STD <= mami <= MAMI_MEAN + MAMI_STD
    return sat


# def wheels_plane_constraint(axiom: str, extra_args: Dict[str, Any]) -> bool:
#     # get low-level blocks (cockpit and wheels)
#     base_position, orientation_forward, orientation_up = Vec.v3i(
#         0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
#     structure = Structure(origin=base_position,
#                           orientation_forward=orientation_forward,
#                           orientation_up=orientation_up)
#     blocks = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
#                               position=base_position).fill_structure(
#                                   structure=structure,
#                                   axiom=axiom,
#                                   additional_args={}).get_all_blocks()
#     cockpit = None
#     wheels = []
#     for b in blocks:
#         if b.block_type == 'LargeBlockCockpit':
#             cockpit = b
#         elif b.block_type == 'OffroadSmallRealWheel1x1':
#             wheels.append(b)
#     # if there are no wheels, the constraint is unsat
#     if len(wheels) == 0:
#         return False
#     else:
#         # create groups of wheels on the same plane
#         wheel_groups = []
#         for wheel in wheels:
#             added = False
#             for group in wheel_groups:
#                 ref_wheel = group["wheels"][0]
#                 ref_x, ref_y, ref_z = ref_wheel.position.as_tuple()
#                 x, y, z = wheel.position.as_tuple()
#                 if group["on"] == "any":
#                     if x == ref_x:
#                         group["on"] = "x"
#                         group["wheels"].append(wheel)
#                         added = True
#                         break
#                     elif y == ref_y:
#                         group["on"] = "y"
#                         group["wheels"].append(wheel)
#                         added = True
#                         break
#                     elif z == ref_z:
#                         group["on"] = "z"
#                         group["wheels"].append(wheel)
#                         added = True
#                         break
#                 elif group["on"] == "x":
#                     if x == ref_x:
#                         group["wheels"].append(wheel)
#                         added = True
#                         break
#                 elif group["on"] == "y":
#                     if y == ref_y:
#                         group["wheels"].append(wheel)
#                         added = True
#                         break
#                 elif group["on"] == "z":
#                     if z == ref_z:
#                         group["wheels"].append(wheel)
#                         added = True
#                         break
#             if not added:
#                 wheel_groups.append({"on": "any", "wheels": [wheel]})
#         # check at least one group of wheels has same orientation UP as cockpit
#         sat = False
#         for group in wheel_groups:
#             sat |= group['wheels'][0].orientation_up == cockpit.orientation_up
#         return sat