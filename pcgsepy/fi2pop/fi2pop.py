from typing import List, Tuple

from ..config import POP_SIZE
from ..evo.fitness import compute_fitness
from ..lsystem.lsystem import LSystem
from .utils import create_new_pool, reduce_population, subdivide_axioms


def fi2pop(f_pop: List[str],
           i_pop: List[str],
           f_fitnesses: List[float],
           i_fitnesses: List[float],
           n_iter: int,
           lsystem: LSystem) -> Tuple[List[str], List[str]]:
    f_pool = []
    i_pool = []

    f_pool_fitnesses = []
    i_pool_fitnesses = []
    for gen in range(n_iter):
        # create offsprings from feasible population
        new_pool = create_new_pool(population=f_pop,
                                   fitnesses=f_fitnesses,
                                   generation=gen)
        # if feasible, add to feasible pool
        # if infeasible, add to infeasible pool
        axioms_sats = subdivide_axioms(hl_axioms=new_pool, lsystem=lsystem)
        for axiom in axioms_sats.keys():
            if axioms_sats[axiom]['feasible']:
                f_pool.append(axiom)
                f_pool_fitnesses.append(
                    compute_fitness(axiom=lsystem.get_ll_axioms(
                        lsystem.get_ml_axioms(hl_axioms=[axiom]))[0][0],
                                    extra_args={
                                        'alphabet':
                                            lsystem.llsolver.atoms_alphabet
                                    }) - axioms_sats[axiom]['n_constraints_v'])
            else:
                i_pool.append(axiom)
                i_pool_fitnesses.append(axioms_sats[axiom]['n_constraints_v'])
        # place the infeasible population in the infeasible pool
        i_pool.extend(i_pop)
        i_pool_fitnesses.extend(i_fitnesses)
        # reduce the infeasible pool if > pops_size
        if len(i_pool) > POP_SIZE:
            i_pool, i_pool_fitnesses = reduce_population(
                population=i_pool, fitnesses=i_pool_fitnesses, to=POP_SIZE)
        # set the infeasible pool as the infeasible population
        i_pop[:] = i_pool[:]
        i_fitnesses[:] = i_pool_fitnesses[:]
        # create offsprings from infeasible population
        new_pool = create_new_pool(population=i_pop,
                                   fitnesses=i_fitnesses,
                                   generation=gen)
        # if feasible, add to feasible pool
        # if infeasible, add to infeasible pool
        axioms_sats = subdivide_axioms(hl_axioms=new_pool, lsystem=lsystem)
        for axiom in axioms_sats.keys():
            if axioms_sats[axiom]['feasible']:
                f_pool.append(axiom)
                f_pool_fitnesses.append(
                    compute_fitness(axiom=lsystem.get_ll_axioms(
                        lsystem.get_ml_axioms(hl_axioms=[axiom]))[0][0],
                                    extra_args={
                                        'alphabet':
                                            lsystem.llsolver.atoms_alphabet
                                    }) - axioms_sats[axiom]['n_constraints_v'])
            else:
                i_pool.append(axiom)
                i_pool_fitnesses.append(axioms_sats[axiom]['n_constraints_v'])
        # reduce the feasible pool if > pops_size
        if len(f_pool) > POP_SIZE:
            f_pool, f_pool_fitnesses = reduce_population(
                population=f_pool, fitnesses=f_pool_fitnesses, to=POP_SIZE)
        # set the feasible pool as the feasible population
        f_pop[:] = f_pool[:]
        f_fitnesses[:] = f_pool_fitnesses[:]

        # highest_if.append(max(f_fitnesses))
        # mean_if.append(sum(f_fitnesses) / len(f_fitnesses))

    return f_pop, i_pop