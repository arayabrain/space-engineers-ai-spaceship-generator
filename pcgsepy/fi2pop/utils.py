from typing import List

from pcgsepy.config import GEN_PATIENCE, MAX_STRING_LEN, POP_SIZE
from pcgsepy.evo.genops import (EvoException, crossover, mutate,
                                roulette_wheel_selection)
from pcgsepy.lsystem.constraints import ConstraintLevel, ConstraintTime
from pcgsepy.lsystem.lsystem import LSystem
from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.structure import IntersectionException


def subdivide_solutions(lcs: List[CandidateSolution],
                        lsystem: LSystem,
                        ) -> None:
    """Assign feasibility flag to the solutions.

    Args:
        lcs (List[CandidateSolution]): The list of solutions.
        lsystem (LSystem): The L-system used to check feasibility for.
    """
    lsystem.hl_solver.set_constraints(cs=lsystem.all_hl_constraints)
    lsystem.ll_solver.set_constraints(cs=lsystem.all_ll_constraints)
    removable = []
    for i, cs in enumerate(lcs):
        try:
            for t in [ConstraintTime.DURING, ConstraintTime.END]:
                sat = lsystem.hl_solver._check_constraints(cs=cs,
                                                           when=t,
                                                           keep_track=True)
                cs.is_feasible &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
                cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
                cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]
            if cs.is_feasible:
                for t in [ConstraintTime.DURING, ConstraintTime.END]:
                    sat = lsystem.ll_solver._check_constraints(cs=cs,
                                                               when=t,
                                                               keep_track=True)
                    cs.is_feasible &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
                    cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
                    cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]
        except IntersectionException:
            pass
        except MemoryError:
            removable.append(i)
    for i in list(reversed(removable)):
        lcs.pop(i)


def create_new_pool(population: List[CandidateSolution],
                    generation: int,
                    n_individuals: int = POP_SIZE,
                    minimize: bool = False) -> List[CandidateSolution]:
    """Create a new pool of solutions.

    Args:
        population (List[CandidateSolution]): Initial population of solutions.
        generation (int): Current generation number.
        n_individuals (int, optional): The number of individuals in the pool. Defaults to POP_SIZE.
        minimize (bool, optional): Whether to minimize or maximize the fitness. Defaults to False.

    Raises:
        EvoException: If same parent is picked twice for crossover.

    Returns:
        List[CandidateSolution]: The pool of new solutions.
    """
    pool = []
    patience = GEN_PATIENCE
    while len(pool) < n_individuals:
        prev_len_pool = len(pool)
        childs = []
        # apply crossover if possible
        if len(population) > 1:
            # fitness-proportionate selection
            p1 = roulette_wheel_selection(pop=population,
                                          minimize=minimize)
            new_pop = []
            new_pop[:] = population[:]
            new_pop.remove(p1)
            p2 = roulette_wheel_selection(pop=new_pop,
                                        minimize=minimize)
            if p1 != p2:
                # crossover
                o1, o2 = crossover(a1=p1, a2=p2, n_childs=2)
                # set parents
                o1.parents = [p1, p2]
                o2.parents = [p1, p2]
                childs = [o1, o2]
            else:
                raise EvoException('Picked same parents, this should never happen.')
        # else, copy twice
        else:
            o1 = CandidateSolution(string=population[0].string[:])
            o2 = CandidateSolution(string=population[0].string[:])
            o1.hls_mod = population[0].hls_mod.copy()
            o2.hls_mod = population[0].hls_mod.copy()
            if population[0].parents:
                o1.parents = population[0].parents
                o2.parents = population[0].parents
                population[0].parents[0].n_offspring += 2
                population[0].parents[1].n_offspring += 2
            childs = [o1, o2]
        for o in childs:
            if MAX_STRING_LEN == -1 or len(o.string) <= MAX_STRING_LEN:
                # mutation
                try:
                    mutate(cs=o, n_iteration=generation)
                except EvoException:
                    pass
                if o not in pool:
                    pool.append(o)
        if len(pool) == prev_len_pool:
            patience -= 1
        else:
            patience = GEN_PATIENCE
        if patience == 0:
            break
    return pool


def reduce_population(population: List[CandidateSolution],
                      to: int,
                      minimize: bool = False) -> List[CandidateSolution]:
    """Order and reduce a population to a given size.

    Args:
        population (List[CandidateSolution]): The population.
        to (int): The desired population size.
        minimize (bool, optional): Whether to order for descending (False) or ascending (True) values. Defaults to False.

    Returns:
        List[CandidateSolution]: The ordered and culled population.
    """
    population.sort(key=lambda x: x.c_fitness, reverse=not minimize)
    return population[:to]
