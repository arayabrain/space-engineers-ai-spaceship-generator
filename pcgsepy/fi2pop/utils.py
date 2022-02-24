from typing import Any, Dict, List, Tuple

from pcgsepy.lsystem.solution import CandidateSolution

from ..config import POP_SIZE
from ..evo.genops import crossover, mutate, roulette_wheel_selection
from ..lsystem.constraints import ConstraintLevel, ConstraintTime
from ..lsystem.lsystem import LSystem
from ..lsystem.parser import HLtoMLTranslator, LLParser


def subdivide_solutions(lcs: List[CandidateSolution],
                        lsystem: LSystem) -> None:
    lsystem.hl_solver.set_constraints(cs=lsystem.all_hl_constraints)
    lsystem.ll_solver.set_constraints(cs=lsystem.all_ll_constraints)
    for cs in lcs:
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            sat = lsystem.hl_solver._check_constraints(cs=cs,
                                                       when=t,
                                                       keep_track=True)
            cs.is_feasible = sat[ConstraintLevel.HARD_CONSTRAINT][0]
            cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
            cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]
        if cs.is_feasible:
            ml_string = lsystem.hl_solver.translator.transform(string=cs.string)
            cs.ll_string = LLParser(rules=lsystem.hl_solver.ll_rules).expand(string=ml_string)
            for t in [ConstraintTime.DURING, ConstraintTime.END]:
                sat = lsystem.ll_solver._check_constraints(cs=cs,
                                                           when=t,
                                                           keep_track=True)
                cs.is_feasible &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
                cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
                cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]


def create_new_pool(population: List[CandidateSolution],
                    generation: int,
                    n_individuals: int = POP_SIZE,
                    minimize: bool = False) -> List[CandidateSolution]:
    pool = []

    while len(pool) < n_individuals:
        # fitness-proportionate selection
        p1 = roulette_wheel_selection(pop=population,
                                      minimize=minimize)
        p2 = roulette_wheel_selection(pop=population,
                                      minimize=minimize)
        # crossover
        o1, o2 = crossover(a1=p1, a2=p2, n_childs=2)

        for o in [o1, o2]:
            # mutation
            mutate(cs=o, n_iteration=generation)
            if o not in pool:
                pool.append(o)

    return pool


def reduce_population(population: List[CandidateSolution],
                      to: int,
                      minimize: bool = False) -> List[CandidateSolution]:
    population.sort(key=lambda x: x.c_fitness if x.is_feasible else x.ncv,
                    reverse=minimize)
    return population[to:]
