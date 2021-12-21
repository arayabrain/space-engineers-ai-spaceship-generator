from typing import Any, Dict, List, Tuple

from ..config import N_ITERATIONS, POP_SIZE, N_RETRIES
from ..evo.fitness import compute_fitness
from ..evo.genops import crossover, mutate, roulette_wheel_selection
from ..lsystem.constraints import ConstraintLevel, ConstraintTime
from ..lsystem.lsystem import LSystem
from ..lsystem.parser import HLtoMLTranslator, LLParser


def subdivide_axioms(hl_axioms: List[str],
                     lsystem: LSystem) -> Dict[str, Any]:
    axioms_sats = {}
    for hl_axiom in hl_axioms:
        axioms_sats[hl_axiom] = {'feasible': True, 'n_constraints_v': 0}
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            sat = lsystem.hlsolver._check_constraints(axiom=hl_axiom,
                                                      when=t,
                                                      keep_track=True)
            axioms_sats[hl_axiom]['feasible'] &= sat[
                ConstraintLevel.HARD_CONSTRAINT][0]
            axioms_sats[hl_axiom]['n_constraints_v'] += sat[
                ConstraintLevel.HARD_CONSTRAINT][1]
            axioms_sats[hl_axiom]['n_constraints_v'] += sat[
                ConstraintLevel.SOFT_CONSTRAINT][1]

    to_expand_further = []
    for hl_axiom in axioms_sats.keys():
        if axioms_sats[hl_axiom]['feasible']:
            to_expand_further.append(hl_axiom)

    ml_axioms = lsystem.get_ml_axioms(hl_axioms=to_expand_further)
    for i, ml_axiom in enumerate(ml_axioms):
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            ll_axiom = LLParser(rules=lsystem.hlsolver.ll_rules).expand(
                axiom=ml_axiom)
            sat = lsystem.llsolver._check_constraints(axiom=ll_axiom,
                                                      when=t,
                                                      keep_track=True)
            axioms_sats[to_expand_further[i]]['feasible'] &= sat[
                ConstraintLevel.HARD_CONSTRAINT][0]
            axioms_sats[to_expand_further[i]]['n_constraints_v'] += sat[
                ConstraintLevel.HARD_CONSTRAINT][1]
            axioms_sats[to_expand_further[i]]['n_constraints_v'] += sat[
                ConstraintLevel.SOFT_CONSTRAINT][1]

    return axioms_sats


def generate_initial_populations(
        lsystem: LSystem,
        pops_size: int = POP_SIZE,
        n_retries: int = N_RETRIES) -> Tuple[List[str], List[str],
                                             List[float], List[float]]:
    feasible_pop, infeasible_pop = [], []
    f_fitnesses, i_fitnesses = [], []
    i = 0
    lsystem.check_sat = False
    while len(feasible_pop) < pops_size or len(infeasible_pop) < pops_size:
        hl_axioms = lsystem.get_hl_axioms(starting_axiom="begin",
                                          iterations=N_ITERATIONS)
        axioms_sats = subdivide_axioms(hl_axioms=hl_axioms, lsystem=lsystem)
        for axiom in axioms_sats.keys():
            if axioms_sats[axiom]['feasible'] and len(
                    feasible_pop) < pops_size and axiom not in feasible_pop:
                feasible_pop.append(axiom)
                f_fitnesses.append(
                    compute_fitness(axiom=lsystem.get_ll_axioms(
                        lsystem.get_ml_axioms(hl_axioms=[axiom]))[0][0],
                                    extra_args={
                                        'alphabet':
                                            lsystem.llsolver.atoms_alphabet
                                    }) - axioms_sats[axiom]['n_constraints_v'])
            elif not axioms_sats[axiom]['feasible'] and len(
                    infeasible_pop) < pops_size and axiom not in feasible_pop:
                infeasible_pop.append(axiom)
                i_fitnesses.append(axioms_sats[axiom]['n_constraints_v'])
        i += 1
        if i == n_retries:
            break
    return feasible_pop, infeasible_pop, f_fitnesses, i_fitnesses


def create_new_pool(population: List[str],
                    fitnesses: List[float],
                    generation: int,
                    translator: HLtoMLTranslator,
                    n_individuals: int = POP_SIZE) -> List[str]:
    pool = []

    while len(pool) < n_individuals:
        # fitness-proportionate selection
        p1 = roulette_wheel_selection(axioms=population, fitnesses=fitnesses)
        p2 = roulette_wheel_selection(axioms=population, fitnesses=fitnesses)
        # crossover
        o1, o2 = crossover(a1=p1, a2=p2, n_childs=2, translator=translator)
        # mutation
        o1 = mutate(axiom=o1, translator=translator, n_iteration=generation)
        o2 = mutate(axiom=o2, translator=translator, n_iteration=generation)

        for o in [o1, o2]:
            if o not in pool:
                pool.append(o)

    return pool


def reduce_population(population: List[str],
                      fitnesses: List[str],
                      to: int) -> Tuple[List[str], List[str]]:
    f_ordered_idxs = [
        i for _, i in sorted(zip(fitnesses, range(len(fitnesses))))
    ][-to:]
    red_pop, red_f = [], []
    for i in f_ordered_idxs:
        red_pop.append(population[i])
        red_f.append(fitnesses[i])
    return red_pop, red_f