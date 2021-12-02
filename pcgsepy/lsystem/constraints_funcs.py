from typing import Any, Dict, List, Tuple
from .constraints import HLStructure
from .actions import AtomAction
from .structure_maker import StructureMaker
from ..common.vecs import Vec


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
