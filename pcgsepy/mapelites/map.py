from tqdm.notebook import trange
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random
import torch as th

from .bin import MAPBin
from .behaviors import BehaviorCharacterization
from ..common.vecs import Orientation, Vec
from ..config import BIN_POP_SIZE, CS_MAX_AGE, N_ITERATIONS, N_RETRIES, POP_SIZE
from ..fi2pop.utils import DimensionalityReducer, MLPEstimator, prepare_dataset, subdivide_solutions, create_new_pool, train_estimator
from ..lsystem.lsystem import LSystem
from ..lsystem.solution import CandidateSolution
from ..lsystem.structure_maker import LLStructureMaker
from ..structure import Structure
from ..evo.fitness import Fitness
from ..evo.genops import EvoException


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
                 behavior_descriptors: Tuple[BehaviorCharacterization, BehaviorCharacterization],
                 n_bins: Tuple[int, int] = (8, 8)):
        self.lsystem = lsystem
        self.feasible_fitnesses = feasible_fitnesses
        self.b_descs = behavior_descriptors
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
        
        self.reducer = DimensionalityReducer(n_components=10,
                                             max_dims=50000)
        self.estimator = MLPEstimator(10, 1)
        self.buffer = {
            'structures': [],
            'xs': [],
            'ys': []
            }

    def show_metric(self,
                    metric: str,
                    show_mean: bool = True,
                    population: str = 'feasible'):
        disp_map = np.zeros(shape=self.bins.shape)
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                disp_map[i, j] = self.bins[i, j].get_metric(metric=metric,
                                                            use_mean=show_mean,
                                                            population=population)
        vmaxs = {
            'fitness': {
                'feasible': 4.5,
                'infeasible': 2
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
                   np.cumsum(self.bin_sizes[0]) + self.limits[0])
        plt.yticks(np.arange(self.bin_qnt[1]),
                   np.cumsum(self.bin_sizes[1]) + self.limits[1])
        plt.xlabel(self.b_descs[0].name)
        plt.ylabel(self.b_descs[1].name)
        plt.title(f'CMAP-Elites {"Avg." if show_mean else ""}{metric} ({population})')
        cbar = plt.colorbar()
        cbar.set_label(f'{"mean" if show_mean else "max"} {metric}',
                       rotation=270)
        plt.show()

    def _set_behavior_descriptors(self, cs: CandidateSolution):
        b0 = self.b_descs[0](cs)
        b1 = self.b_descs[1](cs)
        cs.b_descs = (b0, b1)

    def compute_fitness(self,
                        cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> float:
        if cs.is_feasible:
            return sum([f(cs, extra_args) for f in self.feasible_fitnesses])
        else:
            if self.estimator.is_trained:
                with th.no_grad():
                    x = self.reducer.reduce_dims(cs._content)
                    return self.estimator(th.tensor(x).float()).numpy()[0][0]
            else:
                return np.clip(np.abs(np.random.normal(loc=0, scale=1)), 0, 1)

    def generate_initial_populations(self,
                                     pops_size: int = POP_SIZE,
                                     n_retries: int = N_RETRIES):
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
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                }) + (0.5 - cs.ncv)
                        feasible_pop.append(cs)
                    elif not cs.is_feasible and len(infeasible_pop) < pops_size and cs not in feasible_pop:
                        cs.c_fitness = np.clip(np.abs(np.random.normal(loc=0, scale=1)), 0, 1)  # cs.ncv
                        infeasible_pop.append(cs)
                    if cs._content is None:
                        cs.set_content(get_structure(string=self.lsystem.hl_to_ll(cs=cs).string,
                                                     extra_args={
                                                         'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                         }))
                    self._set_behavior_descriptors(cs=cs)
                    cs.age = CS_MAX_AGE
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
        
        # Initialize buffer
        self.buffer['structures'] = [x._content for x in infeasible_pop]
        # Fit PCA
        self.reducer.fit(self.buffer['structures'])       

    def subdivide_range(self,
                        bin_idx: Tuple[int, int]) -> None:
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
                     lcs: List[CandidateSolution]):
        for cs in lcs:
            b0, b1 = cs.b_descs
            i = np.digitize([b0], np.cumsum([0] + self.bin_sizes[0][:-1]) + self.b_descs[0].bounds[0], right=False)[0] - 1
            j = np.digitize([b1], np.cumsum([0] + self.bin_sizes[1][:-1]) + self.b_descs[1].bounds[0], right=False)[0] - 1
            self.bins[i, j].insert_cs(cs)
            self.bins[i, j].remove_old()

    def _age_bins(self,
                  diff: int = -1):
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                cbin = self.bins[i, j]
                cbin.age(diff=diff)

    def _valid_bins(self):
        valid_bins = []
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                cbin = self.bins[i, j]
                if len(cbin._feasible) > 1 and len(cbin._infeasible) > 1:
                    valid_bins.append(cbin)
        return valid_bins

    def _check_res_trigger(self):
        """
        Trigger a resolution increase if at least 1 bin has reached full
        population capacity for both feasible and infeasible populations.
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
        generated = []
        for pop in populations:
            try:
                new_pool = create_new_pool(population=pop,
                                           generation=gen,
                                           n_individuals=2 * BIN_POP_SIZE,
                                           minimize=False)
                subdivide_solutions(lcs=new_pool,
                                    lsystem=self.lsystem)
                
                
                
                f_pop = [x for x in new_pool if x.is_feasible]
                i_pop = [x for x in new_pool if not x.is_feasible]
                
                for cs in i_pop:
                    if cs._content is None:
                        cs.set_content(get_structure(string=self.lsystem.hl_to_ll(cs=cs).string,
                                                     extra_args={
                                                         'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                         }))
                
                # Update buffer
                self._update_buffer(xs=[],
                                    ys=[],
                                    structures=[cs._content for cs in i_pop])
                # Fit PCA
                self.reducer.fit(self.buffer['structures'])
                xs, ys = prepare_dataset(f_pop=f_pop,
                                         reducer=self.reducer)
                self._update_buffer(xs=xs,
                                    ys=ys,
                                    structures=[])
                if len(self.buffer['xs']) > 0:
                    train_estimator(self.estimator,
                                    xs=self.buffer['xs'],
                                    ys=self.buffer['ys'])
                for i in range(self.bins.shape[0]):
                    for j in range(self.bins.shape[1]):
                        for cs in self.bins[i, j]._infeasible:
                            self.compute_fitness(cs=cs,
                                                 extra_args={
                                                     'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                     })
                
                
                for cs in new_pool:
                    # ensure content is set
                    if cs._content is None:
                        cs.set_content(get_structure(string=self.lsystem.hl_to_ll(cs=cs).string,
                                                    extra_args={
                                                        'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                        }))
                    # assign fitness
                    if cs.is_feasible:
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                }) + (0.5 - cs.ncv)
                    else:
                        # cs.c_fitness = cs.ncv
                        cs.c_fitness = self.compute_fitness(cs=cs,
                                                            extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                })
                    # assign behavior descriptors
                    self._set_behavior_descriptors(cs=cs)
                    # set age
                    cs.age = CS_MAX_AGE
                    
                    generated.append(cs)
            except EvoException:
                pass
        return generated
        
    def rand_step(self,
                  gen: int = 0):
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
                          gen: int = 0):
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
                         n_steps: int = 10):
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
              lcs: Optional[List[CandidateSolution]] = None):
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
        if lcs:
            self._update_bins(lcs=lcs)
            self._check_res_trigger()
        else:
            self.generate_initial_populations()

    def get_elite(self,
                  bin_idx: Tuple[int, int],
                  pop: str):
        i, j = bin_idx
        chosen_bin = self.bins[i, j]
        return chosen_bin.get_elite(population=pop)

    def non_empty(self,
                  bin_idx: Tuple[int, int],
                  pop: str):
        i, j = bin_idx
        chosen_bin = self.bins[i, j]
        pop = chosen_bin._feasible if pop == 'feasible' else chosen_bin._infeasible
        return len(pop) > 0

    def update_behavior_descriptors(self,
                                    bs: Tuple[BehaviorCharacterization]):
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
                                 module: str):
        # toggle module's mutability in the solution
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                self.bins[i, j].toggle_module_mutability(module=module)
        # toggle module's mutability within the L-system
        ms = [x.name for x in self.lsystem.modules]
        self.lsystem.modules[ms.index(module)].active = not self.lsystem.modules[ms.index(module)].active

    def update_fitness_weights(self,
                               weights: List[float]) -> None:
        assert len(weights) == len(self.feasible_fitnesses), f'Wrong number of weights ({len(weights)}) for fitnesses ({len(self.feasible_fitnesses)}) passed.'
        # update weights
        for w, f in zip(weights, self.feasible_fitnesses):
            f.weight = w
        # update solutions fitnesses
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                for cs in self.bins[i, j]._feasible:
                    cs.c_fitness = self.compute_fitness(cs=cs,
                                                        extra_args={
                                                                'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                                }) + (0.5 - cs.ncv)
    
    def _update_buffer(self,
                       xs,
                       ys,
                       structures):
        for x, y in zip(xs, ys):
            if x in self.buffer['xs']:
                i = self.buffer['xs'].index(x)
                curr_y = self.buffer['ys'][i]
                self.buffer['ys'][i] = (y + curr_y) / 2
            else:
                self.buffer['xs'].append(x)
                self.buffer['ys'].append(y)
        for s in structures:
            self.buffer['structures'].append(s)
    
    def shadow_steps(self,
                     gen: int = 0,
                     n_steps: int = 5):
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
