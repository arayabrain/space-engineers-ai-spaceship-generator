from typing import Any, Dict, List, Tuple

from ..config import POP_SIZE
from ..evo.genops import crossover, mutate, roulette_wheel_selection
from ..lsystem.constraints import ConstraintLevel, ConstraintTime
from ..lsystem.lsystem import LSystem
from ..lsystem.parser import HLtoMLTranslator, LLParser


def subdivide_axioms(hl_axioms: List[str], lsystem: LSystem) -> Dict[str, Any]:
    lsystem.hl_solver.set_constraints(cs=lsystem.all_hl_constraints)
    lsystem.ll_solver.set_constraints(cs=lsystem.all_ll_constraints)
    axioms_sats = {}
    for hl_axiom in hl_axioms:
        axioms_sats[hl_axiom] = {
            'feasible': True,
            'n_constraints_v': 0
        }
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            sat = lsystem.hl_solver._check_constraints(axiom=hl_axiom,
                                                       when=t,
                                                       keep_track=True)
            axioms_sats[hl_axiom]['feasible'] &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
            axioms_sats[hl_axiom]['n_constraints_v'] += sat[ConstraintLevel.HARD_CONSTRAINT][1]
            axioms_sats[hl_axiom]['n_constraints_v'] += sat[ConstraintLevel.SOFT_CONSTRAINT][1]

    to_expand_further = []
    for hl_axiom in axioms_sats.keys():
        if axioms_sats[hl_axiom]['feasible']:
            to_expand_further.append(hl_axiom)

    ml_axioms = [lsystem.hl_solver.translator.transform(axiom=hl_axiom) for hl_axiom in to_expand_further]
    for i, ml_axiom in enumerate(ml_axioms):
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            ll_axiom = LLParser(rules=lsystem.hl_solver.ll_rules).expand(axiom=ml_axiom)
            sat = lsystem.ll_solver._check_constraints(axiom=ll_axiom,
                                                       when=t,
                                                       keep_track=True)
            axioms_sats[to_expand_further[i]]['feasible'] &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
            axioms_sats[to_expand_further[i]]['n_constraints_v'] += sat[ConstraintLevel.HARD_CONSTRAINT][1]
            axioms_sats[to_expand_further[i]]['n_constraints_v'] += sat[ConstraintLevel.SOFT_CONSTRAINT][1]

    return axioms_sats


def create_new_pool(population: List[str],
                    fitnesses: List[float],
                    generation: int,
                    translator: HLtoMLTranslator,
                    n_individuals: int = POP_SIZE,
                    minimize: bool = False) -> List[str]:
    pool = []

    while len(pool) < n_individuals:
        # fitness-proportionate selection
        p1 = roulette_wheel_selection(axioms=population,
                                      fitnesses=fitnesses,
                                      minimize=minimize)
        p2 = roulette_wheel_selection(axioms=population,
                                      fitnesses=fitnesses,
                                      minimize=minimize)
        # crossover
        o1, o2 = crossover(a1=p1, a2=p2, n_childs=2)
        # mutation
        o1 = mutate(axiom=o1, n_iteration=generation)
        o2 = mutate(axiom=o2, n_iteration=generation)

        for o in [o1, o2]:
            if o not in pool:
                pool.append(o)

    return pool


def reduce_population(population: List[str],
                      fitnesses: List[str],
                      to: int,
                      minimize: bool = False) -> Tuple[List[str], List[str]]:
    order_on = sorted(zip(fitnesses, range(len(fitnesses))))
    if minimize:
        order_on = reversed(order_on)
    f_ordered_idxs = [i for _, i in order_on][-to:]
    red_pop, red_f = [], []
    for i in f_ordered_idxs:
        red_pop.append(population[i])
        red_f.append(fitnesses[i])
    return red_pop, red_f
