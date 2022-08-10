from typing import List, Tuple

from ..config import (EPSILON_F, GEN_PATIENCE, MAX_STRING_LEN, POP_SIZE,
                      RESCALE_INFEAS_FITNESS)
from ..evo.genops import (EvoException, crossover, mutate,
                          roulette_wheel_selection)
from ..lsystem.constraints import ConstraintLevel, ConstraintTime
from ..lsystem.lsystem import LSystem
from ..lsystem.parser import LLParser
from ..lsystem.solution import CandidateSolution


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
    too_big_cs = []
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
                if cs.ll_string == '':
                    ml_string = lsystem.hl_solver.translator.transform(string=cs.string)
                    cs.ll_string = LLParser(rules=lsystem.hl_solver.ll_rules).expand(string=ml_string)
                for t in [ConstraintTime.DURING, ConstraintTime.END]:
                    sat = lsystem.ll_solver._check_constraints(cs=cs,
                                                            when=t,
                                                            keep_track=True)
                    cs.is_feasible &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
                    cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
                    cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]
        except Exception:
            too_big_cs.append(i)
    for i in list(reversed(too_big_cs)):
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
            for o in [o1, o2]:
                if MAX_STRING_LEN == -1 or len(o.string) <= MAX_STRING_LEN:
                    # mutation
                    try:
                        mutate(cs=o, n_iteration=generation)
                    except EvoException:
                        pass
                    if o not in pool:
                        pool.append(o)       
        else:
            raise EvoException('Picked same parents, this should never happen.')
        if len(pool) == prev_len_pool:
            patience -= 1
        else:
            patience = GEN_PATIENCE
        if patience == 0:
            # print(f'New Pool creation ran out of patience (current pool size: {len(pool)}.')
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
    population.sort(key=lambda x: x.c_fitness,
                    reverse=not minimize)
    return population[:to]


def prepare_dataset(f_pop: List[CandidateSolution]) -> Tuple[List[List[float]]]:
    """Prepare the dataset for the estimator.

    Args:
        f_pop (List[CandidateSolution]): The Feasible population.

    Returns:
        Tuple[List[List[float]]]: Inputs and labels to use during training.
    """
    xs, ys = [], []
    for cs in f_pop:
        y = cs.c_fitness
        for parent in cs.parents:
            if not parent.is_feasible:
                x = parent.representation
                parent.n_feas_offspring += 1
                xs.append(x)
                ys.append(y if not RESCALE_INFEAS_FITNESS else y * (EPSILON_F + (parent.n_feas_offspring / parent.n_offspring)))
    return xs, ys
