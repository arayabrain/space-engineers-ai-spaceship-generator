import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Tuple
from tqdm.notebook import trange

from ..config import POP_SIZE, N_GENS, N_ITERATIONS, N_RETRIES
from ..lsystem.lsystem import LSystem
from .utils import create_new_pool, reduce_population, subdivide_axioms
from ..lsystem.constraints import ConstraintLevel


class FI2PopSolver:
    def __init__(self,
                 feasible_fitnesses: List[Callable[[str, Dict[str, Any]], float]],
                 lsystem: LSystem):
        self.feasible_fitnesses = feasible_fitnesses
        self.lsystem = lsystem
        self.ftop = []
        self.itop = []
        self.fmean = []
        self.imean = []
        # number of total soft constraints
        self.nsc = [c for c in self.lsystem.all_hl_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = [c for c in self.lsystem.all_ll_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = len(self.nsc)

    def _compute_fitness(self,
                         axiom: str,
                         extra_args: Dict[str, Any]) -> float:
        return sum([f(axiom, extra_args) for f in self.feasible_fitnesses])

    def _generate_initial_populations(self,
                                      pops_size: int = POP_SIZE,
                                      n_retries: int = N_RETRIES) -> Tuple[List[str], List[str], List[float], List[float]]:
        feasible_pop, infeasible_pop = [], []
        f_fitnesses, i_fitnesses = [], []
        self.lsystem.disable_sat_check()
        with trange(n_retries, desc='Initialization ') as iterations:
            for i in iterations:
                _, hl_axioms, _ = self.lsystem.apply_rules(starting_axioms=['head', 'body', 'tail'],
                                                           iterations=[1, N_ITERATIONS, 1],
                                                           create_structures=False,
                                                           make_graph=False)
                axioms_sats = subdivide_axioms(hl_axioms=hl_axioms,
                                               lsystem=self.lsystem)
                for axiom in axioms_sats.keys():
                    if axioms_sats[axiom]['feasible'] and len(feasible_pop) < pops_size and axiom not in feasible_pop:
                        feasible_pop.append(axiom)
                        f_fitnesses.append(
                            self._compute_fitness(axiom=self.lsystem.hl_to_ll(axiom=axiom),
                                                  extra_args={
                                                      'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                      }
                                                 )
                            + (self.nsc - axioms_sats[axiom]['n_constraints_v']))
                    elif not axioms_sats[axiom]['feasible'] and len(infeasible_pop) < pops_size and axiom not in feasible_pop:
                        infeasible_pop.append(axiom)
                        i_fitnesses.append(axioms_sats[axiom]['n_constraints_v'])
                iterations.set_postfix(ordered_dict={'fpop-size': f'{len(feasible_pop)}/{pops_size}',
                                                     'ipop-size': f'{len(infeasible_pop)}/{pops_size}'},
                                       refresh=True)
                if i == n_retries or (len(feasible_pop) == pops_size and len(infeasible_pop) == pops_size):
                    break
        return feasible_pop, infeasible_pop, f_fitnesses, i_fitnesses

    def initialize(self) -> Tuple[List[str], List[str], List[float], List[float]]:
        f_pop, i_pop, f_fitnesses, i_fitnesses = self._generate_initial_populations()
        self.ftop.append(max(f_fitnesses))
        self.fmean.append(sum(f_fitnesses) / len(f_fitnesses))
        self.itop.append(min(i_fitnesses))
        self.imean.append(sum(i_fitnesses) / len(i_fitnesses))
        print(f'Created Feasible population of size {len(f_pop)}: t:{self.ftop[-1]};m:{self.fmean[-1]}')
        print(f'Created Infeasible population of size {len(i_pop)}: t:{self.itop[-1]};m:{self.imean[-1]}')
        return f_pop, i_pop, f_fitnesses, i_fitnesses

    def fi2pop(self,
               f_pop: List[str],
               i_pop: List[str],
               f_fitnesses: List[float],
               i_fitnesses: List[float],
               n_iter: int = N_GENS) -> Tuple[List[str], List[str], List[float], List[float]]:
        f_pool = []
        i_pool = []
        f_pool_fitnesses = []
        i_pool_fitnesses = []
        with trange(n_iter, desc='Generation ') as gens:
            for gen in gens:
                # create offsprings from feasible population
                new_pool = create_new_pool(population=f_pop,
                                           fitnesses=f_fitnesses,
                                           generation=gen,
                                           translator=self.lsystem.hl_solver.translator)
                # if feasible, add to feasible pool
                # if infeasible, add to infeasible pool
                axioms_sats = subdivide_axioms(hl_axioms=new_pool,
                                               lsystem=self.lsystem)
                for axiom in axioms_sats.keys():
                    if axioms_sats[axiom]['feasible']:
                        f_pool.append(axiom)
                        f_pool_fitnesses.append(self._compute_fitness(axiom=self.lsystem.hl_to_ll(axiom=axiom),
                                                                      extra_args={
                                                                          'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                          }
                                                                      )
                                                + (self.nsc - axioms_sats[axiom]['n_constraints_v']))
                    else:
                        i_pool.append(axiom)
                        i_pool_fitnesses.append(axioms_sats[axiom]['n_constraints_v'])
                # place the infeasible population in the infeasible pool
                i_pool.extend(i_pop)
                i_pool_fitnesses.extend(i_fitnesses)
                # reduce the infeasible pool if > pops_size
                if len(i_pool) > POP_SIZE:
                    i_pool, i_pool_fitnesses = reduce_population(population=i_pool,
                                                                 fitnesses=i_pool_fitnesses,
                                                                 to=POP_SIZE,
                                                                 minimize=True)
                # set the infeasible pool as the infeasible population
                i_pop[:] = i_pool[:]
                i_fitnesses[:] = i_pool_fitnesses[:]
                # create offsprings from infeasible population
                new_pool = create_new_pool(population=i_pop,
                                           fitnesses=i_fitnesses,
                                           generation=gen,
                                           translator=self.lsystem.hl_solver.translator,
                                           minimize=True)
                # if feasible, add to feasible pool
                # if infeasible, add to infeasible pool
                axioms_sats = subdivide_axioms(hl_axioms=new_pool,
                                               lsystem=self.lsystem)
                for axiom in axioms_sats.keys():
                    if axioms_sats[axiom]['feasible']:
                        f_pool.append(axiom)
                        f_pool_fitnesses.append(self._compute_fitness(axiom=self.lsystem.hl_to_ll(axiom=axiom),
                                                                      extra_args={
                                                                          'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                          }
                                                                      )
                                                + (self.nsc - axioms_sats[axiom]['n_constraints_v']))
                    else:
                        i_pool.append(axiom)
                        i_pool_fitnesses.append(axioms_sats[axiom]['n_constraints_v'])
                # reduce the feasible pool if > pops_size
                if len(f_pool) > POP_SIZE:
                    f_pool, f_pool_fitnesses = reduce_population(population=f_pool,
                                                                 fitnesses=f_pool_fitnesses,
                                                                 to=POP_SIZE)
                # set the feasible pool as the feasible population
                f_pop[:] = f_pool[:]
                f_fitnesses[:] = f_pool_fitnesses[:]
                # update tracking
                self.ftop.append(max(f_fitnesses))
                self.fmean.append(sum(f_fitnesses) / len(f_fitnesses))
                self.itop.append(min(i_fitnesses))
                self.imean.append(sum(i_fitnesses) / len(i_fitnesses))
                gens.set_postfix(ordered_dict={'top-f': self.ftop[-1],
                                               'mean-f': self.fmean[-1],
                                               'top-i': self.itop[-1],
                                               'mean-i': self.imean[-1]},
                                 refresh=True)

        return f_pop, i_pop, f_fitnesses, i_fitnesses

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