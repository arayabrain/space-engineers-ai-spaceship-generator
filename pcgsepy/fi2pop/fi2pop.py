import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from tqdm.notebook import trange

from ..lsystem.solution import CandidateSolution
from ..config import POP_SIZE, N_GENS, N_ITERATIONS, N_RETRIES
from ..lsystem.lsystem import LSystem
from .utils import create_new_pool, reduce_population, subdivide_solutions
from ..lsystem.constraints import ConstraintLevel
from ..evo.fitness import Fitness


class FI2PopSolver:
    def __init__(self,
                 feasible_fitnesses: List[Fitness],
                 lsystem: LSystem):
        self.feasible_fitnesses = feasible_fitnesses
        self.lsystem = lsystem
        self.ftop = []
        self.itop = []
        self.fmean = []
        self.imean = []

        self.ffs, self.ifs = [], []

        # number of total soft constraints
        self.nsc = [c for c in self.lsystem.all_hl_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = [c for c in self.lsystem.all_ll_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = len(self.nsc)

    def _compute_fitness(self,
                         cs: CandidateSolution,
                         extra_args: Dict[str, Any]) -> float:
        return sum([f(cs, extra_args) for f in self.feasible_fitnesses])

    def _generate_initial_populations(self,
                                      pops_size: int = POP_SIZE,
                                      n_retries: int = N_RETRIES) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
        feasible_pop, infeasible_pop = [], []
        self.lsystem.disable_sat_check()
        with trange(n_retries, desc='Initialization ') as iterations:
            for i in iterations:
                solutions = self.lsystem.apply_rules(starting_strings=['head', 'body', 'tail'],
                                                     iterations=[1, N_ITERATIONS, 1],
                                                     create_structures=False,
                                                     make_graph=False)
                subdivide_solutions(lcs=solutions,
                                    lsystem=self.lsystem)
                for cs in solutions:
                    if cs.is_feasible and len(feasible_pop) < pops_size and cs not in feasible_pop:
                        feasible_pop.append(cs)
                        cs.c_fitness = self._compute_fitness(cs=cs,
                                                             extra_args={
                                                                 'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                 }) + (self.nsc - cs.ncv)
                    elif not cs.is_feasible and len(infeasible_pop) < pops_size and cs not in feasible_pop:
                        infeasible_pop.append(cs)
                iterations.set_postfix(ordered_dict={'fpop-size': f'{len(feasible_pop)}/{pops_size}',
                                                     'ipop-size': f'{len(infeasible_pop)}/{pops_size}'},
                                       refresh=True)
                if i == n_retries or (len(feasible_pop) == pops_size and len(infeasible_pop) == pops_size):
                    break
        return feasible_pop, infeasible_pop

    def initialize(self) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
        f_pop, i_pop = self._generate_initial_populations()
        f_fitnesses = [cs.c_fitness for cs in f_pop]
        i_fitnesses = [cs.ncv for cs in i_pop]
        self.ftop.append(max(f_fitnesses))
        self.fmean.append(sum(f_fitnesses) / len(f_fitnesses))
        self.itop.append(min(i_fitnesses))
        self.imean.append(sum(i_fitnesses) / len(i_fitnesses))
        self.ffs.append([self.ftop[-1][0], self.fmean[-1][0]])
        self.ifs.append([self.itop[-1], self.imean[-1]])
        print(f'Created Feasible population of size {len(f_pop)}: t:{self.ftop[-1]};m:{self.fmean[-1]}')
        print(f'Created Infeasible population of size {len(i_pop)}: t:{self.itop[-1]};m:{self.imean[-1]}')
        return f_pop, i_pop

    def fi2pop(self,
               f_pop: List[CandidateSolution],
               i_pop: List[CandidateSolution],
               n_iter: int = N_GENS) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
        f_pool = []
        i_pool = []
        with trange(n_iter, desc='Generation ') as gens:
            for gen in gens:
                # create offsprings from feasible population
                new_pool = create_new_pool(population=f_pop,
                                           generation=gen)
                # if feasible, add to feasible pool
                # if infeasible, add to infeasible pool
                subdivide_solutions(lcs=new_pool,
                                    lsystem=self.lsystem)
                for cs in new_pool:
                    if cs.is_feasible:
                        f_pool.append(cs)
                        cs.ll_string = self.lsystem.hl_to_ll(cs=cs).string
                        cs.c_fitness = self._compute_fitness(cs=cs,
                                                             extra_args={
                                                                 'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                 }
                                                             ) + (self.nsc - cs.ncv)
                    else:
                        i_pool.append(cs)
                # place the infeasible population in the infeasible pool
                i_pool.extend(i_pop)
                # reduce the infeasible pool if > pops_size
                if len(i_pool) > POP_SIZE:
                    i_pool = reduce_population(population=i_pool,
                                               to=POP_SIZE,
                                               minimize=True)
                # set the infeasible pool as the infeasible population
                i_pop[:] = i_pool[:]
                # create offsprings from infeasible population
                new_pool = create_new_pool(population=i_pop,
                                           generation=gen,
                                           minimize=True)
                # if feasible, add to feasible pool
                # if infeasible, add to infeasible pool
                subdivide_solutions(lcs=new_pool,
                                    lsystem=self.lsystem)
                for cs in new_pool:
                    if cs.is_feasible:
                        f_pool.append(cs)
                        cs.ll_string = self.lsystem.hl_to_ll(cs=cs).string
                        cs.c_fitness = self._compute_fitness(cs=cs,
                                                             extra_args={
                                                                 'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                 }
                                                             ) + (self.nsc - cs.ncv)
                    else:
                        i_pool.append(cs)
                # reduce the feasible pool if > pops_size
                if len(f_pool) > POP_SIZE:
                    f_pool = reduce_population(population=f_pool,
                                               to=POP_SIZE)
                # set the feasible pool as the feasible population
                f_pop[:] = f_pool[:]
                # update tracking
                f_fitnesses = [cs.c_fitness for cs in f_pop]
                i_fitnesses = [cs.ncv for cs in i_pop]
                self.ftop.append(max(f_fitnesses))
                self.fmean.append(sum(f_fitnesses) / len(f_fitnesses))
                self.itop.append(min(i_fitnesses))
                self.imean.append(sum(i_fitnesses) / len(i_fitnesses))
                self.ffs.append([self.ftop[-1][0], self.fmean[-1][0]])
                self.ifs.append([self.itop[-1], self.imean[-1]])
                gens.set_postfix(ordered_dict={'top-f': self.ftop[-1],
                                               'mean-f': self.fmean[-1],
                                               'top-i': self.itop[-1],
                                               'mean-i': self.imean[-1]},
                                 refresh=True)

        return f_pop, i_pop

    def plot_trackings(self,
                       title: str) -> None:
        for t, m, pop in zip((self.ftop, self.itop), (self.fmean, self.imean), ('Feasible', 'Infeasible')):
            plt.plot(range(len(t)), t, label=f'Top {pop}', c='#4CD7D0', lw=2)
            plt.plot(range(len(m)), m, label=f'Mean {pop}', c='#4C5270', lw=2)
            plt.legend()
            plt.title(f'{title} for {pop} population')
            plt.ylabel('Fitness')
            plt.xlabel('Generations')
            plt.savefig(f'lsystem-fi2pop-{pop[0].lower()}-fitnesses.png', transparent=True)
            plt.show()