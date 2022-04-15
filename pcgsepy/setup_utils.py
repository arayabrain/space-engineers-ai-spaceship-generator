import json
from typing import List

import matplotlib
from matplotlib import pyplot as plt

from pcgsepy.common.vecs import Vec, orientation_from_str
from pcgsepy.config import COMMON_ATOMS, HL_ATOMS
from pcgsepy.lsystem.actions import AtomAction, Rotations
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
    if type3_fix:
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
    if larger_fonts:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 26

        PAD_TITLE_SIZE = 20
        PAD_LABEL_SIZE = 10

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_default_lsystem(used_ll_blocks: List[str]) -> LSystem:
    with open(COMMON_ATOMS, "r") as f:
        common_alphabet = json.load(f)

        for k in common_alphabet:
            action, args = common_alphabet[k]["action"], common_alphabet[k]["args"]
            action = AtomAction(action)
            if action == AtomAction.MOVE:
                args = orientation_from_str[args]
            elif action == AtomAction.ROTATE:
                args = Rotations(args)
            common_alphabet[k] = {"action": action, "args": args}

    with open(HL_ATOMS, "r") as f:
        hl_atoms = json.load(f)

    tiles_dimensions = {}
    tiles_block_offset = {}
    for tile in hl_atoms.keys():
        dx, dy, dz = hl_atoms[tile]["dimensions"]
        tiles_dimensions[tile] = Vec.v3i(dx, dy, dz)
        tiles_block_offset[tile] = hl_atoms[tile]["offset"]

    hl_alphabet = {}
    for k in common_alphabet.keys():
        hl_alphabet[k] = common_alphabet[k]

    for hk in hl_atoms.keys():
        hl_alphabet[hk] = {"action": AtomAction.PLACE, "args": []}

    ll_alphabet = {}

    for k in common_alphabet.keys():
        ll_alphabet[k] = common_alphabet[k]
    for k in used_ll_blocks:
        ll_alphabet[k] = {"action": AtomAction.PLACE, "args": [k]}

    hl_rules = RuleMaker(ruleset='hlrules').get_rules()
    ll_rules = RuleMaker(ruleset='llrules').get_rules()

    hl_parser = HLParser(rules=hl_rules)
    ll_parser = LLParser(rules=ll_rules)

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

    lsystem = LSystem(
        hl_solver=hl_solver, ll_solver=ll_solver, names=[
            'HeadModule', 'BodyModule', 'TailModule']
    )

    lsystem.add_hl_constraints(cs=[
        [nic, rcc1],
        [nic, rcc2],
        [nic, rcc3]
    ])

    lsystem.add_ll_constraints(cs=[
        [sc],
        [sc],
        [sc]
    ])

    return lsystem
