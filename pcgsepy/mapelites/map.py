import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from pcgsepy.lsystem.constraints import ConstraintLevel
from pcgsepy.mapelites.bandit import EpsilonGreedyAgent
from pcgsepy.mapelites.buffer import Buffer, EmptyBufferException
from pcgsepy.mapelites.emitters import (ContextualBanditEmitter, Emitter,
                                        HumanPrefMatrixEmitter, RandomEmitter,
                                        emitters, get_emitter_by_str)
from pcgsepy.nn.estimators import QuantileEstimator
from tqdm.notebook import trange
from typing_extensions import Self

from ..common.vecs import Orientation, Vec
from ..config import (ALIGNMENT_INTERVAL, BIN_POP_SIZE, CS_MAX_AGE, EPSILON_F,
                      MAX_X_SIZE, MAX_Y_SIZE, MAX_Z_SIZE, N_ITERATIONS,
                      N_RETRIES, POP_SIZE)
from ..evo.fitness import (Fitness, box_filling_fitness, func_blocks_fitness,
                           mame_fitness, mami_fitness)
from ..evo.genops import EvoException
from ..fi2pop.utils import (GaussianEstimator, create_new_pool,
                            prepare_dataset, subdivide_solutions)
from ..hullbuilder import HullBuilder
from ..lsystem.lsystem import LSystem
from ..lsystem.solution import CandidateSolution
from ..lsystem.structure_maker import LLStructureMaker
from ..nn.estimators import MLPEstimator, QuantileEstimator, train_estimator
from ..structure import Structure
from .behaviors import BehaviorCharacterization
from .bin import MAPBin


# TEMPORARY, CandidateSolution content should be set earlier
def get_structure(string: str,
                  extra_args: Dict[str, Any]):
    base_position, orientation_forward, orientation_up = Vec.v3i(0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                 position=base_position).fill_structure(structure=structure,
                                                                        string=string,
                                                                        additional_args={})
    structure.update(origin=base_position,
                     orientation_forward=orientation_forward,
                     orientation_up=orientation_up)
    return structure

def coverage_reward(mapelites: 'MAPElites') -> float:
    tot_coverage, inc_coverage = mapelites.bins.shape[0] * mapelites.bins.shape[1], 0.
    for map_bin in mapelites.bins.flatten().tolist():
        is_new_bin = False
        if len(map_bin._feasible) > 0:
            for cs in map_bin._feasible:
                if cs.age != CS_MAX_AGE:
                    is_new_bin = False
        inc_coverage += 1 if is_new_bin else 0
    return inc_coverage / tot_coverage

def fitness_reward(mapelites: 'MAPElites') -> float:
    prev_best, current_best = 0, 0
    for map_bin in mapelites.bins.flatten().tolist():
        if len(map_bin._feasible) > 0:
            for cs in map_bin._feasible:
                if cs.age == CS_MAX_AGE:
                    if cs.c_fitness > prev_best:
                        current_best = cs.c_fitness
                elif cs.age != CS_MAX_AGE:
                    if cs.c_fitness > prev_best:
                        prev_best = cs.c_fitness
    fit_diff = current_best - prev_best
    return fit_diff / prev_best


agent_rewards = {
    'coverage_reward': coverage_reward,
    'fitness_reward': fitness_reward
}


class MAPElites:

    def __init__(self,
                 lsystem: LSystem,
                 feasible_fitnesses: List[Fitness],
                 buffer: Buffer,
                 behavior_descriptors: Tuple[BehaviorCharacterization, BehaviorCharacterization],
                 n_bins: Tuple[int, int] = (8, 8),
                 estimator: Optional[Union[GaussianEstimator, MLPEstimator, QuantileEstimator]] = None,  # Only MLPEstimator is currently JSON-compatible
                 emitter: Optional[Emitter] = RandomEmitter(),
                 agent: Optional[EpsilonGreedyAgent] = None,
                 agent_rewards: Optional[List[Callable[[Self], float]]] = []):
        """Create a MAP-Elites object.

        Args:
            lsystem (LSystem): The L-system used to expand strings.
            feasible_fitnesses (List[Fitness]): The list of fitnesses used.
            behavior_descriptors (Tuple[BehaviorCharacterization, BehaviorCharacterization]): The X- and Y-axis behavior descriptors.
            n_bins (Tuple[int, int], optional): The number of X and Y bins. Defaults to (8, 8).
        """
        self.lsystem = lsystem
        # number of total soft constraints
        self.nsc = [c for c in self.lsystem.all_hl_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = [c for c in self.lsystem.all_ll_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = len(self.nsc) * 0.5
        self.feasible_fitnesses = feasible_fitnesses
        self.b_descs = behavior_descriptors
        self.emitter = emitter
        self.agent = agent
        self.agent_rewards = agent_rewards
        self.limits = (self.b_descs[0].bounds[1] if self.b_descs[0].bounds is not None else 20,
                       self.b_descs[1].bounds[1] if self.b_descs[1].bounds is not None else 20)
        self._initial_n_bins = n_bins
        self.bin_qnt = n_bins
        self.bin_sizes = [
            [self.limits[0] / self.bin_qnt[0]] * n_bins[0],
            [self.limits[1] / self.bin_qnt[1]] * n_bins[1]
        ]

        self.bins = np.empty(shape=self.bin_qnt, dtype=object)
        for i in range(self.bin_qnt[0]):
            for j in range(self.bin_qnt[1]):
                self.bins[i, j] = MAPBin(bin_idx=(i, j),
                                         bin_size=(self.bin_sizes[0][i],
                                                   self.bin_sizes[1][j]))
        self.enforce_qnt = True
        self.hull_builder = HullBuilder(erosion_type='bin',
                                        apply_erosion=True,
                                        apply_smoothing=False)
        
        self.estimator = estimator
        self.buffer = buffer
        
        self.max_f_fitness = sum([f.bounds[1] for f in self.feasible_fitnesses])
        self.max_i_fitness = len(self.lsystem.all_hl_constraints) if not self.estimator else 1
        self.infeas_fitness_idx = 1  # 0: min, 1: median, 2: max
        
        self.allow_res_increase = True
        self.allow_aging = True
                
        assert self.agent is not None or self.emitter is not None, 'MAP-Elites requires either an agent or an emitter!'
        if self.agent is not None and not self.agent_rewards: raise AssertionError(f'You selected an agent but no reward functions have been provided!')

    def show_metric(self,
                    metric: str,
                    show_mean: bool = True,
                    population: str = 'feasible',
                    save_as: str = '') -> None:
        """Show the bin metric.

        Args:
            metric (str): The metric to display.
            show_mean (bool, optional): Whether to show average metric or elite's. Defaults to True.
            population (str, optional): Which population to show metric of. Defaults to 'feasible'.
        """
        disp_map = np.zeros(shape=self.bins.shape)
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                disp_map[i, j] = self.bins[i, j].get_metric(metric=metric,
                                                            use_mean=show_mean,
                                                            population=population)
        vmaxs = {
            'fitness': {
                'feasible': self.max_f_fitness,
                'infeasible': self.max_i_fitness
            },
            'age': {
                'feasible': CS_MAX_AGE,
                'infeasible': CS_MAX_AGE
            },
            'size': {
                'feasible': BIN_POP_SIZE,
                'infeasible': BIN_POP_SIZE
            }
        }
        plt.imshow(disp_map,
                   origin='lower',
                   cmap='hot',
                   interpolation='nearest',
                   vmin=0,
                   vmax=vmaxs[metric][population])
        plt.xticks(np.arange(self.bin_qnt[0]),
                   np.cumsum(self.bin_sizes[0]) + self.b_descs[0].bounds[0])
        plt.yticks(np.arange(self.bin_qnt[1]),
                   np.cumsum(self.bin_sizes[1]) + self.b_descs[1].bounds[0])
        plt.xlabel(self.b_descs[1].name)
        plt.ylabel(self.b_descs[0].name)
        plt.title(f'CMAP-Elites {"Avg." if show_mean else ""}{metric} ({population})')
        cbar = plt.colorbar()
        cbar.set_label(f'{"mean" if show_mean else "max"} {metric}',
                       rotation=270)
        if save_as != '':
            title_part = metric + ('-avg-' if show_mean else '-top-') + population
            plt.savefig(f'results/{save_as}-{title_part}.png', transparent=True, bbox_inches='tight')
        plt.show()

    def _set_behavior_descriptors(self,
                                  cs: CandidateSolution) -> None:
        """Set the behavior descriptors of the solution.

        Args:
            cs (CandidateSolution): The candidate solution.
        """
        b0 = self.b_descs[0](cs)
        b1 = self.b_descs[1](cs)
        cs.b_descs = (b0, b1)

    def compute_fitness(self,
                        cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> float:
        """Compute the fitness of the solution. 

        Args:
            cs (CandidateSolution): The solution.
            extra_args (Dict[str, Any]): Additional arguments used in the fitness computation.

        Returns:
            float: The fitness value.
        """
        if cs.fitness == []:
            if cs.is_feasible:
                cs.fitness = [f(cs, extra_args) for f in self.feasible_fitnesses]
                cs.representation = cs.fitness[:]
                x, y, z = cs.content._max_dims
                cs.representation.extend([x / MAX_X_SIZE,
                                          y / MAX_Y_SIZE,
                                          z / MAX_Z_SIZE])
            else:
                cs.representation = [f(cs, extra_args) for f in [Fitness(name='BoxFilling', f=box_filling_fitness, bounds=(0, 1)),
                                                                 Fitness(name='FuncionalBlocks', f=func_blocks_fitness, bounds=(0, 1)),
                                                                 Fitness(name='MajorMediumProportions', f=mame_fitness, bounds=(0, 1)),
                                                                 Fitness(name='MajorMinimumProportions', f=mami_fitness, bounds=(0, 1))]]
        if cs.is_feasible:
            return sum([self.feasible_fitnesses[i].weight * cs.fitness[i] for i in range(len(cs.fitness))])
        else:
            if self.estimator is not None:
                if isinstance(self.estimator, GaussianEstimator):
                    return self.estimator.predict(x=np.asarray(cs.fitness))
                elif isinstance(self.estimator, MLPEstimator):
                    if self.estimator.is_trained:
                        with th.no_grad():
                            return self.estimator(th.tensor(cs.representation).float()).numpy()[0]
                    else:
                        return EPSILON_F
                elif isinstance(self.estimator, QuantileEstimator):
                    if self.estimator.is_trained:
                        with th.no_grad():
                            # set fitness to (3,) array (min, median, max)
                            cs.fitness = self.estimator(th.tensor(cs.representation).float().unsqueeze(0)).numpy()[0]
                            return cs.fitness[self.infeas_fitness_idx]  # set c_itness to median by default
                    else:
                        cs.fitness = [EPSILON_F, EPSILON_F, EPSILON_F]
                        return cs.fitness[self.infeas_fitness_idx]
                else:
                    raise NotImplementedError(f'Unrecognized estimator type: {type(self.estimator)}.')
            else:
                return cs.ncv

    def generate_initial_populations(self,
                                     pops_size: int = POP_SIZE,
                                     n_retries: int = N_RETRIES) -> None:
        """Generate the initial populations.

        Args:
            pops_size (int, optional): The size of the populations. Defaults to POP_SIZE.
            n_retries (int, optional): The number of initialization retries. Defaults to N_RETRIES.
        """
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
                    if cs._content is None:
                            cs.set_content(get_structure(string=self.lsystem.hl_to_ll(cs=cs).string,
                                                        extra_args={
                                                            'alphabet': self.lsystem.ll_solver.atoms_alphabet
                            }))
                    if cs.is_feasible and len(feasible_pop) < pops_size and cs not in feasible_pop:
                        if self.hull_builder is not None:
                            self.hull_builder.add_external_hull(structure=cs._content)
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                            }) + (self.nsc - cs.ncv)
                        self._set_behavior_descriptors(cs=cs)
                        cs.age = CS_MAX_AGE
                        
                        cs._content = None
                        
                        feasible_pop.append(cs)
                    elif not cs.is_feasible and len(infeasible_pop) < pops_size and cs not in feasible_pop:
                        if self.hull_builder is not None:
                            self.hull_builder.add_external_hull(structure=cs._content)
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                            })
                        self._set_behavior_descriptors(cs=cs)
                        cs.age = CS_MAX_AGE
                        
                        cs._content = None
                        
                        infeasible_pop.append(cs)
                iterations.set_postfix(ordered_dict={
                    'fpop-size': f'{len(feasible_pop)}/{pops_size}',
                    'ipop-size': f'{len(infeasible_pop)}/{pops_size}'
                },
                                       refresh=True)
                if i == n_retries or (len(feasible_pop) == pops_size and len(infeasible_pop) == pops_size):
                    break
        self._update_bins(lcs=feasible_pop)
        self._update_bins(lcs=infeasible_pop)
        if self.emitter is not None and self.emitter.requires_init:
            self.emitter.init_emitter(bins=self.bins)

    def subdivide_range(self,
                        bin_idx: Tuple[int, int]) -> None:
        """Subdivide the chosen bin range.
        For each bin, 4 new bins are created and the solutions are redistributed amongst those.

        Args:
            bin_idx (Tuple[int, int]): The index of the bin.
        """
        i, j = bin_idx
        # get solutions in same row&col
        all_cs = []
        for m in range(self.bins.shape[0]):
            for n in range(self.bins.shape[1]):
                if m == i or n == j:
                    all_cs.extend(self.bins[m, n]._feasible)
                    all_cs.extend(self.bins[m, n]._infeasible)
        # update bin sizes
        v_i, v_j = self.bin_sizes[0][i], self.bin_sizes[1][j]
        self.bin_sizes[0][i] = v_i / 2
        self.bin_sizes[1][j] = v_j / 2
        self.bin_sizes[0].insert(i + 1, v_i / 2)
        self.bin_sizes[1].insert(j + 1, v_j / 2)
        # update bin quantity
        self.bin_qnt = (self.bin_qnt[0] + 1, self.bin_qnt[1] + 1)
        # create new bin map
        new_bins = np.empty(shape=self.bin_qnt, dtype=object)
        # copy over unaffected bins
        new_bins[:i, :j] = self.bins[:i, :j]
        new_bins[:i, (j+2):] = self.bins[:i, (j+1):]
        new_bins[(i+2):, :j] = self.bins[(i+1):, :j]
        new_bins[(i+2):, (j+2):] = self.bins[(i+1):, (j+1):]
        # populate newly created bins
        for m in range(new_bins.shape[0]):
            for n in range(new_bins.shape[1]):
                if m == i or m == i + 1 or n == j or n == j + 1:
                    new_bins[m, n] = MAPBin(bin_idx=(m, n),
                                            bin_size=(self.bin_sizes[0][m],
                                                      self.bin_sizes[1][n]))
                new_bins[m, n].bin_idx = (m, n)
        # assign new bin map
        self.bins = new_bins
        # assign solutions to bins
        self._update_bins(lcs=all_cs)
        if type(self.emitter) is HumanPrefMatrixEmitter:
            self.emitter._increase_preferences_res(idx=bin_idx) 

    def _update_bins(self,
                     lcs: List[CandidateSolution]) -> None:
        """Update the bins by assigning new solutions.

        Args:
            lcs (List[CandidateSolution]): The list of new solutions.
        """
        bc0 = np.cumsum([0] + self.bin_sizes[0][:-1]) + self.b_descs[0].bounds[0]
        bc1 = np.cumsum([0] + self.bin_sizes[1][:-1]) + self.b_descs[1].bounds[0]
        for cs in lcs:
            b0, b1 = cs.b_descs
            i = np.digitize(x=[b0],
                            bins=bc0,
                            right=False)[0] - 1
            j = np.digitize(x=[b1],
                            bins=bc1,
                            right=False)[0] - 1
            self.bins[i, j].insert_cs(cs)
            self.bins[i, j].remove_old()

    def _age_bins(self,
                  diff: int = -1) -> None:
        """Age all bins.

        Args:
            diff (int, optional): The quantity to age for. Defaults to -1.
        """
        if self.allow_aging:
            for i in range(self.bins.shape[0]):
                for j in range(self.bins.shape[1]):
                    cbin = self.bins[i, j]
                    cbin.age(diff=diff)

    def _valid_bins(self) -> List[MAPBin]:
        """Get all the valid bins.
        A valid bin is a bin with at least 1 Feasible solution and 1 Infeasible solution.

        Returns:
            List[MAPBin]: The list of valid bins.
        """
        valid_bins = []
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                cbin = self.bins[i, j]
                if len(cbin._feasible) > 1 and len(cbin._infeasible) > 1:
                    valid_bins.append(cbin)
        return valid_bins

    def _check_res_trigger(self) -> None:
        """
        Trigger a resolution increase if at least 1 bin has reached full
        population capacity for both Feasible and Infeasible populations.
        """
        if self.allow_res_increase:
            to_increase_res = []
            for i in range(self.bins.shape[0]):
                for j in range(self.bins.shape[1]):
                    cbin = self.bins[i, j]
                    if len(cbin._feasible) >= BIN_POP_SIZE and len(cbin._infeasible) >= BIN_POP_SIZE:
                        to_increase_res.append((i, j))
            if to_increase_res:
                for bin_idx in to_increase_res:
                    self.subdivide_range(bin_idx=bin_idx)

    def _step(self,
              populations: List[List[CandidateSolution]],
              gen: int) -> List[CandidateSolution]:
        """Apply a single step of modified FI2Pop.

        Args:
            populations (List[List[CandidateSolution]]): The Feasible and Infeasible populations.
            gen (int): The current generation number.

        Returns:
            List[CandidateSolution]: The list of new solutions.
        """
        generated = []
        for pop in populations:
            if len(pop) > 0:
                try:
                    minimize = False if pop[0].is_feasible else False if self.estimator is not None else True
                    new_pool = create_new_pool(population=pop,
                                            generation=gen,
                                            n_individuals=len(pop),
                                            minimize=minimize)
                    subdivide_solutions(lcs=new_pool,
                                        lsystem=self.lsystem)
                    # ensure content is set
                    for cs in new_pool:
                        if cs._content is None:
                            cs.set_content(get_structure(string=self.lsystem.hl_to_ll(cs=cs).string,
                                                        extra_args={
                                                            'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                            }))
                        # add hull
                        if self.hull_builder is not None:
                            self.hull_builder.add_external_hull(structure=cs._content)
                        # assign fitness
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                            })
                        if cs.is_feasible:
                            cs.c_fitness += (self.nsc - cs.ncv)
                        # assign behavior descriptors
                        self._set_behavior_descriptors(cs=cs)
                        # set age
                        cs.age = CS_MAX_AGE
                        
                        cs._content = None
                        
                        generated.append(cs)
                    
                    if self.estimator is not None:
                        # subdivide for estimator/
                        f_pop = [x for x in new_pool if x.is_feasible]
                        # Prepare dataset for estimator
                        xs, ys = prepare_dataset(f_pop=f_pop)
                        for x, y in zip(xs, ys):
                            self.buffer.insert(x=x,
                                               y=y / self.max_f_fitness)
                        # If possible, train estimator
                        try:
                            xs, ys = self.buffer.get()
                            if isinstance(self.estimator, GaussianEstimator):
                                self.estimator.fit(x=xs,
                                                   y=ys)
                            elif isinstance(self.estimator, MLPEstimator) or isinstance(self.estimator, QuantileEstimator):
                                train_estimator(self.estimator,
                                                xs=xs,
                                                ys=ys)
                            else:
                                raise NotImplementedError(f'Unrecognized estimator type {type(self.estimator)}.')
                        except EmptyBufferException:
                            pass
                        if self.estimator.is_trained and gen % ALIGNMENT_INTERVAL == 0:
                            # Reassign previous infeasible fitnesses
                            for i in range(self.bins.shape[0]):
                                for j in range(self.bins.shape[1]):
                                    for cs in self.bins[i, j]._infeasible:
                                        if cs.age > ALIGNMENT_INTERVAL:
                                            cs.c_fitness = self.compute_fitness(cs=cs,
                                                                                extra_args={
                                                                                    'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                                    })
                except EvoException:
                    pass
        return generated

    def rand_step(self,
                  gen: int = 0) -> None:
        """Apply a random step.

        Args:
            gen (int, optional): The current number of generations. Defaults to 0.
        """
        # trigger aging of solution
        self._age_bins()
        # pick random bin
        rnd_bin = random.choice(self._valid_bins())
        f_pop = rnd_bin._feasible
        i_pop = rnd_bin._infeasible
        generated = self._step(populations=[f_pop, i_pop],
                               gen=gen)
        if generated:
            self._update_bins(lcs=generated)
            self._check_res_trigger()
        else:
            self._age_bins(diff=1)

    def _interactive_step(self,
                          bin_idxs: List[Tuple[int, int]],
                          gen: int = 0) -> None:
        """Applies an interactive step.

        Args:
            bin_idxs (List[Tuple[int, int]]): The indexes of the bins selected.
            gen (int, optional): The current number of generations. Defaults to 0.
        """
        self._age_bins()
        chosen_bins = [self.bins[bin_idx[0], bin_idx[1]] for bin_idx in bin_idxs]
        f_pop, i_pop = [], []
        for chosen_bin in chosen_bins:
            if self.enforce_qnt:
                assert chosen_bin in self._valid_bins(), f'Bin at {chosen_bin.bin_idx} is not a valid bin.'
            f_pop += chosen_bin._feasible
            i_pop += chosen_bin._infeasible
        generated = self._step(populations=[f_pop, i_pop],
                               gen=gen)
        if generated:
            self._update_bins(lcs=generated)
            self._check_res_trigger()
        else:
            self._age_bins(diff=1)
        if self.emitter is not None and self.emitter.requires_pre:
            self.emitter.pre_step(bins=self.bins,
                                  selected_idxs=bin_idxs)

    def interactive_mode(self,
                         n_steps: int = 10) -> None:
        """Start an interactive evolution session. Bins choice is done via `input`.

        Args:
            n_steps (int, optional): The number of steps to evolve for. Defaults to 10.
        """
        for n in range(n_steps):
            print(f'### STEP {n+1}/{n_steps} ###')
            valid_bins = self._valid_bins()
            list_valid_bins = '\n-' + '\n-'.join([str(x.bin_idx) for x in valid_bins])
            print(f'Valid bins are: {list_valid_bins}')
            chosen_bin = None
            while chosen_bin is None:
                choice = input('Enter valid bin to evolve: ')
                choice = choice.replace(' ', '').split(',')
                selected = self.bins[int(choice[0]), int(choice[1])]
                if selected in valid_bins:
                    chosen_bin = selected
                else:
                    print('Chosen bin is not amongst valid bins.')
            self._interactive_step(bin_idxs=[chosen_bin.bin_idx],
                                   gen=n)

    def reset(self,
              lcs: Optional[List[CandidateSolution]] = None) -> None:
        """Reset the current MAP-Elites.

        Args:
            lcs (Optional[List[CandidateSolution]], optional): If provided, the solutions are assigned to the new MAP-Elites. Defaults to None.
        """
        self.bin_qnt = self._initial_n_bins
        self.bin_sizes = [
            [self.limits[0] / self.bin_qnt[0]] * self._initial_n_bins[0],
            [self.limits[1] / self.bin_qnt[1]] * self._initial_n_bins[1]
        ]

        self.bins = np.empty(shape=self.bin_qnt, dtype=object)
        for i in range(self.bin_qnt[0]):
            for j in range(self.bin_qnt[1]):
                self.bins[i, j] = MAPBin(bin_idx=(i, j),
                                         bin_size=(self.bin_sizes[0][i],
                                                   self.bin_sizes[1][j]))

        if self.estimator is not None:
            if isinstance(self.estimator, MLPEstimator):
                self.estimator = MLPEstimator(xshape=self.estimator.xshape,
                                            yshape=self.estimator.yshape)
            elif isinstance(self.estimator, QuantileEstimator):
                self.estimator = QuantileEstimator(xshape=self.estimator.xshape,
                                                   yshape=self.estimator.yshape)
        self.buffer.clear()

        if self.emitter is not None:
            self.emitter.reset()
        
        if lcs is not None:
            self._update_bins(lcs=lcs)
            self._check_res_trigger()
        else:
            self.generate_initial_populations()

    def get_elite(self,
                  bin_idx: Tuple[int, int],
                  pop: str) -> CandidateSolution:
        """Get the elite solution at the selected bin.

        Args:
            bin_idx (Tuple[int, int]): The index of the bin.
            pop (str): The population.

        Returns:
            CandidateSolution: The elite solution.
        """
        i, j = bin_idx
        chosen_bin = self.bins[i, j]
        return chosen_bin.get_elite(population=pop)

    def update_behavior_descriptors(self,
                                    bs: Tuple[BehaviorCharacterization]) -> None:
        """Update the behavior descriptors used in the MAP-Elites.

        Args:
            bs (Tuple[BehaviorCharacterization]): The 2 new behavior descriptors.
        """
        self.b_descs = bs
        self.limits = (self.b_descs[0].bounds[1] if self.b_descs[0].bounds is not None else 20,
                       self.b_descs[1].bounds[1] if self.b_descs[1].bounds is not None else 20)
        lcs = []
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                lcs.extend(self.bins[i, j]._feasible)
                lcs.extend(self.bins[i, j]._infeasible)
        for cs in lcs:
            self._set_behavior_descriptors(cs=cs)
        self.reset(lcs=lcs)

    def toggle_module_mutability(self,
                                 module: str) -> None:
        """Toggle the mutability of the module.

        Args:
            module (str): The module's name.
        """
        # toggle module's mutability in the solution
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                self.bins[i, j].toggle_module_mutability(module=module)
        # toggle module's mutability within the L-system
        ms = [x.name for x in self.lsystem.modules]
        self.lsystem.modules[ms.index(module)].active = not self.lsystem.modules[ms.index(module)].active

    def update_fitness_weights(self,
                               weights: List[float]) -> None:
        """Update the weights of the Feasible fitnesses.

        Args:
            weights (List[float]): The new list of weights.
        """
        assert len(weights) == len(self.feasible_fitnesses), f'Wrong number of weights ({len(weights)}) for fitnesses ({len(self.feasible_fitnesses)}) passed.'
        # update weights
        for w, f in zip(weights, self.feasible_fitnesses):
            f.weight = w
        # update solutions fitnesses
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                for cs in self.bins[i, j]._feasible:
                    cs.c_fitness = sum([self.feasible_fitnesses[i].weight * cs.fitness[i] for i in range(len(cs.fitness))])
                    cs.c_fitness += (self.nsc - cs.ncv)

    def compute_bandit_reward(self) -> float:
        return sum([f(self) for f in self.agent_rewards])
    
    def emitter_step(self,
                     gen: int = 0) -> None:
        """Apply a step according to the emitter.

        Args:
            gen (int, optional): The current generation number. Defaults to 0.
        """
        emitter, bandit = None, None
        if self.emitter is not None:
            emitter = self.emitter
        elif self.agent is not None:
            if type(self.agent) is EpsilonGreedyAgent:
                # get bandit
                bandit = self.agent.choose_bandit()
                emitter_str, method_str = bandit.action.split(';')
                # set emitter
                emitter = get_emitter_by_str(emitter=emitter_str)
                # set merge method
                if method_str == 'max':
                    self.infeas_fitness_idx = 0
                elif method_str == 'median':
                    self.infeas_fitness_idx = 1
                elif method_str == 'min':
                    self.infeas_fitness_idx = 2
                else:
                    raise NotImplementedError(f'Unrecognized merge method from bandit action: {method_str}')
                # update existing solution's fitness
                for i in range(self.bins.shape[0]):
                    for j in range(self.bins.shape[1]):
                        for cs in self.bins[i, j]._infeasible:
                            cs.c_fitness = cs.fitness[self.infeas_fitness_idx]
            elif type(self.agent) is ContextualBanditEmitter:
                pass
        else:
            raise NotImplementedError('MAP-Elites requires either a fixed emitter or a MultiArmed Bandit Agent, but neither were provided.')
        selected_bins = emitter.pick_bin(bins=self.bins)
        fpop, ipop = [], []
        if isinstance(selected_bins[0], MAPBin):
            for selected_bin in selected_bins:
                fpop.extend(selected_bin._feasible)
                ipop.extend(selected_bin._infeasible)
        elif isinstance(selected_bins[0], list):
            for selected_bin in selected_bins[0]:
                fpop.extend(selected_bin._feasible)
            for selected_bin in selected_bins[1]:
                ipop.extend(selected_bin._infeasible)
        else:
            raise NotImplementedError(f'Unrecognized emitter output: {selected_bins}.')
        generated = self._step(populations=[fpop, ipop],
                               gen=gen)
        if generated:
            self._update_bins(lcs=generated)
            self._check_res_trigger()
        if self.emitter is not None and self.emitter.requires_post:
            self.emitter.post_step(bins=self.bins)
        if bandit is not None:
            r = self.compute_bandit_reward()
            self.agent.reward_bandit(bandit=bandit,
                                     reward=r)
    
    def get_coverage(self,
                     pop: str) -> Tuple[int, int]:
        """Get the grid coverage.

        Args:
            pop (str): The population.

        Returns:
            Tuple[int, int]: The number of non-empty bins and the total number of bins.
        """
        c, t = 0, 0
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                if self.bins[i, j].non_empty(pop=pop):
                    c += 1
                t += 1
        return c, t
    
    def get_fitness_metrics(self,
                            pop: str) -> Tuple[int, int]:
        """Get the fitness metrics of a population.

        Args:
            pop (str): The population.

        Returns:
            Tuple[int, int]: The top and mean fitness.
        """
        fs = []
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                p = self.bins[i, j]._feasible if pop == 'feasible' else self.bins[i, j]._infeasible
                for cs in p:
                    fs.append(cs.c_fitness)
        top = np.max(fs)
        if pop == 'infeasible' and self.estimator is None:
            top = np.min(fs)
        return top, np.average(fs)

    def get_qdscore(self,
                    pop: str) -> float:
        qd = 0
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                if self.bins[i, j].non_empty(pop=pop):
                    qd += self.bins[i, j].get_elite(population=pop).c_fitness
        return qd
    
    def get_new_feas_with_unfeas_parents(self):
        n_new = 0
        total = 0
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                for cs in self.bins[i, j]._feasible:
                    if cs.age == CS_MAX_AGE:
                        if cs.parents:
                            if not cs.parents[0].is_feasible:
                                n_new += 1
                                total += 1
                                break
                for cs in self.bins[i, j]._infeasible:
                    if cs.age == CS_MAX_AGE:
                        total += 1
        return n_new, total
    
    def get_random_elite(self,
                         pop: str) -> CandidateSolution:
        nonempty = []
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                if self.bins[i, j].non_empty(pop=pop):
                    nonempty.append(self.bins[i, j])
        return np.random.choice(nonempty).get_elite(population=pop)

    def to_json(self) -> Dict[str, Any]:
        return {
            'lsystem': self.lsystem.to_json(),
            'feasible_fitnesses': [f.to_json() for f in self.feasible_fitnesses],
            'b_descs': [bd.to_json() for bd in list(self.b_descs)],
            'emitter': self.emitter.to_json() if self.emitter else None,
            'agent': self.agent.to_json() if self.agent else None,
            'agent_rewards': [ar.__name__ for ar in self.agent_rewards],
            'initial_n_bins': list(self._initial_n_bins),
            'bin_qnt': list(self.bin_qnt),
            'bins': [b.to_json() for b in self.bins.flatten().tolist()],
            'enforce_qnt': self.enforce_qnt,
            'estimator': self.estimator.to_json() if self.estimator else None,
            'buffer': self.buffer.to_json(),
            'allow_res_increase': self.allow_res_increase,
            'allow_aging': self.allow_aging,
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'MAPElites':
        me = MAPElites(lsystem=LSystem.from_json(my_args['lsystem']),
                       feasible_fitnesses=[Fitness.from_json(f) for f in my_args['feasible_fitnesses']],
                       buffer=Buffer.from_json(my_args['buffer']),
                       behavior_descriptors=tuple([BehaviorCharacterization.from_json(bc) for bc in my_args['b_descs']]),
                       n_bins=tuple(my_args['bin_qnt']),
                       estimator=None,
                       emitter=RandomEmitter(),
                       agent=None,
                       agent_rewards=None)
        me._initial_n_bins = my_args['initial_n_bins']
        me.enforce_qnt = my_args['enforce_qnt']
        me.allow_res_increase = my_args['allow_res_increase']
        me.allow_aging = my_args['allow_aging']
        if my_args['emitter']:
            me.emitter = emitters[my_args['emitter']['name']].from_json(my_args['emitter'])
        if my_args['estimator']:
            me.estimator = MLPEstimator.from_json(my_args['estimator'])
        if my_args['agent']:
            me.agent = EpsilonGreedyAgent.from_json(my_args['agent'])
            me.agent_rewards = [agent_rewards[ar] for ar in my_args['agent_rewards']]
        me.bins = np.asarray([MAPBin.from_json(mb) for mb in my_args['bins']]).reshape(me.bin_qnt)
        return me
