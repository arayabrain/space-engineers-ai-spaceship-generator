import copy
import json
from typing import Any, Dict, List

import matplotlib
from matplotlib import pyplot as plt

from pcgsepy.common.vecs import Vec, orientation_from_str
from pcgsepy.config import COMMON_ATOMS, HL_ATOMS
from pcgsepy.lsystem.actions import AtomAction, rotations_from_str
from pcgsepy.lsystem.constraints import (ConstraintHandler, ConstraintLevel,
                                         ConstraintTime)
from pcgsepy.lsystem.constraints_funcs import (components_constraint,
                                               intersection_constraint,
                                               symmetry_constraint)
from pcgsepy.lsystem.lsystem import LSystem
from pcgsepy.lsystem.parser import HLParser, LLParser
from pcgsepy.lsystem.rules import RuleMaker
from pcgsepy.lsystem.solver import LSolver


def setup_matplotlib(type3_fix: bool = True,
                     larger_fonts: bool = True):
    """Setup Matplotlib.

    Args:
        type3_fix (bool, optional): Ensures no Type3 font is used. Defaults to `True`.
        larger_fonts (bool, optional): Enlarge font sizes. Defaults to `True`.
    """
    if type3_fix:
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
    if larger_fonts:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_default_lsystem(used_ll_blocks: List[str]) -> LSystem:
    """Get the default L-system.

    Args:
        used_ll_blocks (List[str]): List of game blocks used.

    Returns:
        LSystem: The default L-system.
    """
    # load the common atoms
    with open(COMMON_ATOMS, "r") as f:
        common_alphabet: Dict[str, Any] = json.load(f)
    # add actions
    action_to_args = {
        AtomAction.MOVE: orientation_from_str,
        AtomAction.ROTATE: rotations_from_str,
        AtomAction.POP: {},
        AtomAction.PUSH: {},
    }
    common_alphabet.update({k: {'action': AtomAction(common_alphabet[k]["action"]),
                                'args': action_to_args[AtomAction(common_alphabet[k]["action"])].get(str(common_alphabet[k]["args"]), common_alphabet[k]["args"])} for k in common_alphabet})
    # load high-level atoms
    with open(HL_ATOMS, "r") as f:
        hl_atoms = json.load(f)
    # get the tile properties
    tiles_dimensions = {tile: Vec.from_tuple(hl_atoms[tile]["dimensions"]) for tile in hl_atoms}
    tiles_block_offset = {tile: hl_atoms[tile]["offset"] for tile in hl_atoms}
    # create the high-level alphabet
    hl_alphabet = {k: v for k, v in common_alphabet.items()}
    hl_alphabet.update({k: {"action": AtomAction.PLACE, "args": []}
                       for k in hl_atoms})
    # create the low-level alphabet
    ll_alphabet = {k: v for k, v in common_alphabet.items()}
    ll_alphabet.update({k: {"action": AtomAction.PLACE, "args": [k]}
                       for k in used_ll_blocks})
    # create the rulesets
    hl_rules = RuleMaker(ruleset='hlrules_sm').get_rules()
    ll_rules = RuleMaker(ruleset='llrules').get_rules()
    # create the parsers
    hl_parser = HLParser(rules=hl_rules)
    ll_parser = LLParser(rules=ll_rules)
    # create the solver
    hl_solver = LSolver(parser=hl_parser,
                        atoms_alphabet=hl_alphabet,
                        extra_args={
                            'tiles_dimensions': tiles_dimensions,
                            'tiles_block_offset': tiles_block_offset,
                            'll_rules': ll_rules
                        })
    ll_solver = LSolver(parser=ll_parser,
                        atoms_alphabet=dict(hl_alphabet, **ll_alphabet),
                        extra_args={})
    # create the constraints
    rcc1 = ConstraintHandler(
        name="required_components",
        level=ConstraintLevel.HARD_CONSTRAINT,
        when=ConstraintTime.END,
        f=components_constraint,
        extra_args={
            'alphabet': hl_alphabet
        }
    )
    rcc1.extra_args["req_tiles"] = ['cockpit']
    rcc2 = ConstraintHandler(
        name="required_components",
        level=ConstraintLevel.HARD_CONSTRAINT,
        when=ConstraintTime.END,
        f=components_constraint,
        extra_args={
            'alphabet': hl_alphabet
        }
    )
    rcc2.extra_args["req_tiles"] = [
        'corridorcargo', 'corridorgyros', 'corridorreactors']
    rcc3 = ConstraintHandler(
        name="required_components",
        level=ConstraintLevel.HARD_CONSTRAINT,
        when=ConstraintTime.END,
        f=components_constraint,
        extra_args={
            'alphabet': hl_alphabet
        }
    )
    rcc3.extra_args["req_tiles"] = ['thrusters']
    nic = ConstraintHandler(
        name="no_intersections",
        level=ConstraintLevel.HARD_CONSTRAINT,
        when=ConstraintTime.DURING,
        f=intersection_constraint,
        extra_args={
            'alphabet': dict(hl_alphabet, **ll_alphabet)
        },
        needs_ll=True
    )
    nic.extra_args["tiles_dimensions"] = tiles_dimensions
    sc = ConstraintHandler(
        name="symmetry",
        level=ConstraintLevel.SOFT_CONSTRAINT,
        when=ConstraintTime.END,
        f=symmetry_constraint,
        extra_args={
            'alphabet': dict(hl_alphabet, **ll_alphabet)
        }
    )
    # create the L-system
    lsystem = LSystem(hl_solver=hl_solver,
                      ll_solver=ll_solver,
                      names=['HeadModule', 'BodyModule', 'TailModule']
                      )
    # add the high-level constraints
    lsystem.add_hl_constraints(cs=[
        [nic, rcc1],
        [nic, rcc2],
        [nic, rcc3]
    ])
    # add the low-level constraints
    lsystem.add_ll_constraints(cs=[
        [sc],
        [sc],
        [sc]
    ])

    return lsystem
