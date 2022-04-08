import random
from typing import Any, Dict, List, Optional, Tuple, Union
from itsdangerous import NoneAlgorithm

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import NotFittedError
import torch as th
from tqdm.notebook import trange
from sklearn.gaussian_process import GaussianProcessRegressor
from pcgsepy.lsystem.constraints import ConstraintLevel
from pcgsepy.mapelites.buffer import Buffer, EmptyBufferException

from pcgsepy.mapelites.emitters import Emitter, RandomEmitter

from ..common.vecs import Orientation, Vec
from ..config import (BIN_POP_SIZE, CS_MAX_AGE, MAX_DIMS_RED, N_DIM_RED,
                      N_ITERATIONS, N_RETRIES, POP_SIZE)
from ..evo.fitness import Fitness
from ..evo.genops import EvoException
from ..fi2pop.utils import (DimensionalityReducer, MLPEstimator,
                            create_new_pool, prepare_dataset,
                            subdivide_solutions, train_estimator)
from ..hullbuilder import HullBuilder
from ..lsystem.lsystem import LSystem
from ..lsystem.solution import CandidateSolution
from ..lsystem.structure_maker import LLStructureMaker
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


class MAPElites:

    def __init__(self,
                 lsystem: LSystem,
                 feasible_fitnesses: List[Fitness],
                 estimator: Union[MLPEstimator, GaussianProcessRegressor],
                 buffer: Buffer,
                 behavior_descriptors: Tuple[BehaviorCharacterization, BehaviorCharacterization],
                 n_bins: Tuple[int, int] = (8, 8),
                 emitter: Emitter = RandomEmitter()):
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
        
        # self.estimator = MLPEstimator(N_DIM_RED, 1)
        self.estimator = estimator
        self.buffer = buffer

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
                'feasible': sum([f.bounds[1] for f in self.feasible_fitnesses]),
                'infeasible': len(self.lsystem.all_hl_constraints)
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
        plt.xlabel(self.b_descs[0].name)
        plt.ylabel(self.b_descs[1].name)
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
        fs = [f(cs, extra_args) for f in self.feasible_fitnesses]
        cs.fitness = fs
        if cs.is_feasible:
            return sum([self.feasible_fitnesses[i].weight * cs.fitness[i] for i in range(len(cs.fitness))])
        else:
            if isinstance(self.estimator, MLPEstimator):
                if self.estimator.is_trained:
                    with th.no_grad():
                        return self.estimator(th.tensor(cs.fitness).float()).numpy()[0]
                else:
                    return np.clip(np.abs(np.random.normal(loc=0, scale=1)), 0, 1)
            elif isinstance(self.estimator, GaussianProcessRegressor):
                try:
                    y_mean = self.estimator.predict(np.asarray(cs.fitness).reshape(1, -1))[0]
                    if y_mean < 0:
                        y_mean = np.max(y_mean, 0)
                    return y_mean
                except NotFittedError:
                    return np.clip(np.abs(np.random.normal(loc=0, scale=1)), 0, 1)
            else:
                return np.clip(np.abs(np.random.normal(loc=0, scale=1)), 0, 1)

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
                        
                        self.hull_builder.add_external_hull(structure=cs._content)
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                            }) + (self.nsc - cs.ncv)
                        self._set_behavior_descriptors(cs=cs)
                        cs.age = CS_MAX_AGE
                        feasible_pop.append(cs)
                    elif not cs.is_feasible and len(infeasible_pop) < pops_size and cs not in feasible_pop:
                        self.hull_builder.add_external_hull(structure=cs._content)
                        cs.fitness = np.clip(np.abs(np.random.normal(loc=0, scale=1, size=len(self.feasible_fitnesses))), 0, 1)
                        cs.c_fitness = np.sum(cs.fitness)
                        self._set_behavior_descriptors(cs=cs)
                        cs.age = CS_MAX_AGE
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
        # ensure it can start
        # if len(self._valid_bins()) == 0:
        #     self.generate_initial_populations(pops_size=pops_size,
        #                                       n_retries=n_retries)

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
        # assign new bin map
        self.bins = new_bins
        # assign solutions to bins
        self._update_bins(lcs=all_cs)

    def _update_bins(self,
                     lcs: List[CandidateSolution]) -> None:
        """Update the bins by assigning new solutions.

        Args:
            lcs (List[CandidateSolution]): The list of new solutions.
        """
        for cs in lcs:
            b0, b1 = cs.b_descs
            i = np.digitize(x=[b0],
                            bins=np.cumsum([0] + self.bin_sizes[0][:-1]) + self.b_descs[0].bounds[0],
                            right=False)[0] - 1
            j = np.digitize(x=[b1],
                            bins=np.cumsum([0] + self.bin_sizes[1][:-1]) + self.b_descs[1].bounds[0],
                            right=False)[0] - 1
            self.bins[i, j].insert_cs(cs)
            self.bins[i, j].remove_old()

    def _age_bins(self,
                  diff: int = -1) -> None:
        """Age all bins.

        Args:
            diff (int, optional): The quantity to age for. Defaults to -1.
        """
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
            try:
                new_pool = create_new_pool(population=pop,
                                           generation=gen,
                                           n_individuals=2 * BIN_POP_SIZE,
                                           minimize=False)
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
                    generated.append(cs)
                
                # subdivide for estimator/pca stuff
                f_pop = [x for x in new_pool if x.is_feasible]
                # Prepare dataset for estimator
                xs, ys = prepare_dataset(f_pop=f_pop)
                for x, y in zip(xs, ys):
                    self.buffer.insert(x=x,
                                       y=y)
                # If possible, train estimator
                try:
                    xs, ys = self.buffer.get()
                    if isinstance(self.estimator, MLPEstimator):
                        train_estimator(self.estimator,
                                        xs=xs,
                                        ys=ys)
                    elif isinstance(self.estimator, GaussianProcessRegressor):
                        self.estimator.fit(xs, ys)
                    # Reassign previous infeasible fitnesses
                    for i in range(self.bins.shape[0]):
                        for j in range(self.bins.shape[1]):
                            for cs in self.bins[i, j]._infeasible:
                                cs.c_fitness = self.compute_fitness(cs=cs,
                                                                    extra_args={
                                                                        'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                        })
                except EmptyBufferException:
                    pass
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

        if isinstance(self.estimator, MLPEstimator):
            self.estimator = MLPEstimator(xshape=self.estimator.xshape,
                                          yshape=self.estimator.yshape)
        elif isinstance(self.estimator, GaussianProcessRegressor):
            self.estimator = GaussianProcessRegressor(kernel=self.estimator.kernel,
                                                      alpha=self.estimator.alpha,
                                                      optimizer=self.estimator.optimizer,
                                                      random_state=self.estimator.random_state)
        self.buffer = Buffer(merge_method=self.buffer._merge)

        if lcs:
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

    def non_empty(self,
                  bin_idx: Tuple[int, int],
                  pop: str) -> bool:
        """Check if the selected bin is not empty for the population.

        Args:
            bin_idx (Tuple[int, int]): The bin index.
            pop (str): The population.

        Returns:
            bool: Whether the bin's population has at least a solution.
        """
        i, j = bin_idx
        chosen_bin = self.bins[i, j]
        pop = chosen_bin._feasible if pop == 'feasible' else chosen_bin._infeasible
        return len(pop) > 0

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
                    

    # def _update_buffer(self,
    #                    xs: List[List[float]],
    #                    ys: List[float],
    #                    structures: List[Structure]) -> None:
    #     """Update the buffer of data points and structures.

    #     Args:
    #         xs (List[List[float]]): The list of X data points (low-dimensional representation of structures).
    #         ys (List[float]): The list of Y data points (average offsprings' fitness).
    #         structures (List[Structure]): The list of structures.
    #     """
    #     for x, y in zip(xs, ys):
    #         if x in self.buffer['xs']:
    #             i = self.buffer['xs'].index(x)
    #             curr_y = self.buffer['ys'][i]
    #             self.buffer['ys'][i] = (y + curr_y) / 2
    #         else:
    #             self.buffer['xs'].append(x)
    #             self.buffer['ys'].append(y)
    #     for s in structures:
    #         self.buffer['structures'].append(s)

    def shadow_steps(self,
                     gen: int = 0,
                     n_steps: int = 5) -> None:
        """Apply some hidden steps.

        Args:
            gen (int, optional): The current number of generations. Defaults to 0.
            n_steps (int, optional): The number of hidden steps to apply. Defaults to 5.
        """
        for n in range(n_steps):
            # find best feas and infeas populations
            feas, infeas = [0, None], [0, None]
            for i in range(self.bins.shape[0]):
                for j in range(self.bins.shape[1]):
                    mf = self.bins[i, j].get_metric(metric='fitness',
                                                    use_mean=True,
                                                    population='feasible')
                    mi = self.bins[i, j].get_metric(metric='fitness',
                                                    use_mean=True,
                                                    population='infeasible')
                    if mf > feas[0]:
                        feas = [mf, self.bins[i, j]]
                    if mi > infeas[0]:
                        infeas = [mi, self.bins[i, j]]
            f_pop = feas[1]._feasible
            i_pop = infeas[1]._infeasible
            generated = self._step(populations=[f_pop, i_pop],
                                   gen=gen)
            if generated:
                self._update_bins(lcs=generated)
                self._check_res_trigger()

    def emitter_step(self,
                     gen: int = 0) -> None:
        """Apply a step according to the emitter.

        Args:
            gen (int, optional): The current generation number. Defaults to 0.
        """
        all_bins = self.bins.flatten().tolist()
        selected_bins = self.emitter.pick_bin(bins=[b for b in all_bins if len(b._feasible) > 0])
        fpop, ipop = [], []
        for selected_bin in selected_bins:
            fpop.extend(selected_bin._feasible)
            ipop.extend(selected_bin._infeasible)
        generated = self._step(populations=[fpop, ipop],
                               gen=gen)
        if generated:
            self._update_bins(lcs=generated)
            self._check_res_trigger()
    
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
                if self.non_empty(bin_idx=(i, j),
                                  pop=pop):
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
        return np.max(fs), np.average(fs)
