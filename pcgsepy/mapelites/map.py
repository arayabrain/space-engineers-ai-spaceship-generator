from tqdm.notebook import trange
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random

from .bin import MAPBin
from ..common.vecs import Orientation, Vec
from ..config import BIN_POP_SIZE, CS_MAX_AGE, N_ITERATIONS
from ..fi2pop.utils import subdivide_axioms, create_new_pool
from ..lsystem.lsystem import LSystem
from ..lsystem.solution import CandidateSolution
from ..lsystem.structure_maker import LLStructureMaker
from ..structure import Structure


# TEMPORARY, CandidateSolution content should be set earlier
def get_structure(axiom: str, extra_args: Dict[str, Any]):
    base_position, orientation_forward, orientation_up = Vec.v3i(
        0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                 position=base_position).fill_structure(
                                     structure=structure,
                                     axiom=axiom,
                                     additional_args={})
    structure.update(origin=base_position,
                     orientation_forward=orientation_forward,
                     orientation_up=orientation_up)
    return structure


class MAPElites:

    def __init__(self,
                 lsystem: LSystem,
                 feasible_fitnesses: List[callable],
                 behavior_limits: Tuple[int, int] = (20, 20),
                 n_bins: Tuple[int, int] = (8, 8)):
        self.lsystem = lsystem
        self.feasible_fitnesses = feasible_fitnesses
        self.limits = behavior_limits
        self.bin_qnt = n_bins
        self.bin_sizes = (self.limits[0] / self.bin_qnt[0],
                          self.limits[1] / self.bin_qnt[1])

        self.bins = np.empty(shape=self.bin_qnt, dtype=object)
        for i in range(self.bin_qnt[0]):
            for j in range(self.bin_qnt[1]):
                self.bins[i, j] = MAPBin(bin_idx=(i, j),
                                         bin_size=self.bin_sizes)

    def show_fitness(self,
                     show_mean: bool = True,
                     population: str = 'feasible'):
        disp_map = np.zeros(shape=self.bins.shape)
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                disp_map[i, j] = self.bins[i,
                                           j].get_metric(metric='fitness',
                                                         use_mean=show_mean,
                                                         population=population)
        plt.imshow(disp_map,
                   origin='lower',
                   cmap='hot',
                   interpolation='nearest',
                   vmin=0,
                   vmax=4.5 if population == 'feasible' else 2)
        plt.xticks(np.arange(self.bin_qnt[0]),
                   np.arange(0, self.limits[0], self.bin_sizes[0]))
        plt.yticks(np.arange(self.bin_qnt[1]),
                   np.arange(0, self.limits[1], self.bin_sizes[1]))
        plt.xlabel('Largest / Smallest')
        plt.ylabel('Largest / Medium')
        plt.title(f'CMAP-Elites ({population})')
        cbar = plt.colorbar()
        cbar.set_label(f'Cumulative fitness ({"mean" if show_mean else "max"})',
                       rotation=270)
        plt.show()

    def show_age(self, show_mean: bool = True, population: str = 'feasible'):
        disp_map = np.zeros(shape=self.bins.shape)
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                disp_map[i, j] = self.bins[i,
                                           j].get_metric(metric='age',
                                                         use_mean=show_mean,
                                                         population=population)
        plt.imshow(disp_map,
                   origin='lower',
                   cmap='hot',
                   interpolation='nearest',
                   vmin=0,
                   vmax=CS_MAX_AGE)
        plt.xticks(np.arange(self.bin_qnt[0]),
                   np.arange(0, self.limits[0], self.bin_sizes[0]))
        plt.yticks(np.arange(self.bin_qnt[1]),
                   np.arange(0, self.limits[1], self.bin_sizes[1]))
        plt.xlabel('Largest / Smallest')
        plt.ylabel('Largest / Medium')
        plt.title(f'CMAP-Elites ({population})')
        cbar = plt.colorbar()
        cbar.set_label(f'Solutions age ({"mean" if show_mean else "max"})',
                       rotation=270)
        plt.show()

    def show_coverage(self, population: str = 'feasible'):
        disp_map = np.zeros(shape=self.bins.shape)
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                disp_map[i, j] = self.bins[i,
                                           j].get_metric(metric='size',
                                                         population=population)
        plt.imshow(disp_map,
                   origin='lower',
                   cmap='hot',
                   interpolation='nearest',
                   vmin=0,
                   vmax=BIN_POP_SIZE)
        plt.xticks(np.arange(self.bin_qnt[0]),
                   np.arange(0, self.limits[0], self.bin_sizes[0]))
        plt.yticks(np.arange(self.bin_qnt[1]),
                   np.arange(0, self.limits[1], self.bin_sizes[1]))
        plt.xlabel('Largest / Smallest')
        plt.ylabel('Largest / Medium')
        plt.title(f'CMAP-Elites ({population})')
        cbar = plt.colorbar()
        cbar.set_label('Number of solutions', rotation=270)
        plt.show()

    def _set_behavior_descriptors(self, cs: CandidateSolution):
        volume = cs.content.as_array().shape
        largest_axis, medium_axis, smallest_axis = reversed(sorted(
            list(volume)))
        mame = largest_axis / medium_axis
        mami = largest_axis / smallest_axis
        cs.b_descs = (mame, mami)

    def compute_fitness(self, axiom: str, extra_args: Dict[str, Any]) -> float:
        return sum([f(axiom, extra_args) for f in self.feasible_fitnesses])

    def generate_initial_populations(
        self,
        pops_size: int = 20,
        n_retries: int = 100,
    ):
        feasible_pop, infeasible_pop = [], []
        self.lsystem.disable_sat_check()
        with trange(n_retries, desc='Initialization ') as iterations:
            for i in iterations:
                _, hl_axioms, _ = self.lsystem.apply_rules(
                    starting_axioms=['head', 'body', 'tail'],
                    iterations=[1, N_ITERATIONS, 1],
                    create_structures=False,
                    make_graph=False)
                axioms_sats = subdivide_axioms(hl_axioms=hl_axioms,
                                               lsystem=self.lsystem)
                for axiom in axioms_sats.keys():
                    cs = CandidateSolution(string=axiom)
                    if axioms_sats[axiom]['feasible'] and len(
                            feasible_pop
                    ) < pops_size and axiom not in feasible_pop:
                        cs.c_fitness = self.compute_fitness(
                            axiom=self.lsystem.hl_to_ll(axiom=axiom),
                            extra_args={
                                'alphabet':
                                    self.lsystem.ll_solver.atoms_alphabet
                            }) + (0.5 - axioms_sats[axiom]['n_constraints_v'])
                        cs.is_feasible = True
                        feasible_pop.append(cs)
                    elif not axioms_sats[axiom]['feasible'] and len(
                            infeasible_pop
                    ) < pops_size and axiom not in feasible_pop:
                        cs.c_fitness = axioms_sats[axiom]['n_constraints_v']
                        infeasible_pop.append(cs)
                    cs.set_content(
                        get_structure(
                            axiom=self.lsystem.hl_to_ll(axiom=axiom),
                            extra_args={
                                'alphabet':
                                    self.lsystem.ll_solver.atoms_alphabet
                            }))
                    self._set_behavior_descriptors(cs=cs)
                    cs.age = CS_MAX_AGE
                iterations.set_postfix(ordered_dict={
                    'fpop-size': f'{len(feasible_pop)}/{pops_size}',
                    'ipop-size': f'{len(infeasible_pop)}/{pops_size}'
                },
                                       refresh=True)
                if i == n_retries or (len(feasible_pop) == pops_size and
                                      len(infeasible_pop) == pops_size):
                    break
        self._update_bins(lcs=feasible_pop)
        self._update_bins(lcs=infeasible_pop)
        # ensure it can start
        if len(self._valid_bins()) == 0:
            self.generate_initial_populations(lsystem=self.lsystem,
                                              pops_size=pops_size,
                                              n_retries=n_retries)

    def _increase_resolution(self):
        # get all solutions
        all_cs = []
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                all_cs.extend(self.bins[i, j]._feasible)
                all_cs.extend(self.bins[i, j]._infeasible)
        # double resolution of bins
        self.bins = np.empty(shape=(self.bins.shape[0] * 2,
                                    self.bins.shape[1] * 2),
                             dtype=object)
        self.bin_qnt = (self.bin_qnt[0] * 2, self.bin_qnt[1] * 2)
        self.bin_sizes = (self.limits[0] / self.bin_qnt[0],
                          self.limits[1] / self.bin_qnt[1])
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                self.bins[i, j] = MAPBin(bin_idx=(i, j),
                                         bin_size=self.bin_sizes)
        # assign solutions to bins
        self._update_bins(lcs=all_cs)

    def _update_bins(self, lcs: List[CandidateSolution]):
        for cs in lcs:
            b0, b1 = cs.b_descs
            i, j = int(b0 // self.bin_sizes[0]), int(b1 // self.bin_sizes[1])
            self.bins[i, j].insert_cs(cs)
            self.bins[i, j].remove_old()

    def _age_bins(self):
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                cbin = self.bins[i, j]
                cbin.age_up()

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
        to_increase_res = False
        for i in range(self.bins.shape[0]):
            for j in range(self.bins.shape[1]):
                cbin = self.bins[i, j]
                if len(cbin._feasible) >= BIN_POP_SIZE and len(
                        cbin._infeasible) >= BIN_POP_SIZE:
                    to_increase_res = True
                    break
        if to_increase_res:
            self._increase_resolution()

    def rand_step(self, gen: int = 0):
        # trigger aging of solution
        self._age_bins()
        # pick random bin
        rnd_bin = random.choice(self._valid_bins())
        f_pop = rnd_bin._feasible
        i_pop = rnd_bin._infeasible
        generated = []
        for pop, minimize in zip([f_pop, i_pop], [False, True]):
            strings = [cs.string for cs in pop]
            fitnesses = [cs.c_fitness for cs in pop]
            new_pool = create_new_pool(
                population=strings,
                fitnesses=fitnesses,
                generation=gen,
                translator=self.lsystem.hl_solver.translator,
                n_individuals=2 * BIN_POP_SIZE,
                minimize=minimize)
            axioms_sats = subdivide_axioms(hl_axioms=new_pool,
                                           lsystem=self.lsystem)
            for axiom in axioms_sats.keys():
                cs = CandidateSolution(string=axiom)
                if axioms_sats[axiom]['feasible']:
                    cs.is_feasible = True
                    cs.c_fitness = self.compute_fitness(
                        axiom=self.lsystem.hl_to_ll(axiom=axiom),
                        extra_args={
                            'alphabet': self.lsystem.ll_solver.atoms_alphabet
                        }) + (0.5 - axioms_sats[axiom]['n_constraints_v'])
                else:
                    cs.c_fitness = axioms_sats[axiom]['n_constraints_v']
                cs.set_content(
                    get_structure(axiom=self.lsystem.hl_to_ll(axiom=axiom),
                                  extra_args={
                                      'alphabet':
                                          self.lsystem.ll_solver.atoms_alphabet
                                  }))
                self._set_behavior_descriptors(cs=cs)
                cs.age = CS_MAX_AGE
                generated.append(cs)
        self._update_bins(lcs=generated)
        self._check_res_trigger()

    def interactive_mode(self, n_steps: int = 10):
        for n in range(n_steps):
            print(f'### STEP {n+1}/{n_steps} ###')
            self.show_age(show_mean=True, population='feasible')
            self.show_age(show_mean=True, population='infeasible')

            valid_bins = self._valid_bins()
            list_valid_bins = '\n-' + '\n-'.join(
                [str(x.bin_idx) for x in valid_bins])
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

            f_pop = chosen_bin._feasible
            i_pop = chosen_bin._infeasible
            generated = []
            for pop, minimize in zip([f_pop, i_pop], [False, True]):
                strings = [cs.string for cs in pop]
                fitnesses = [cs.c_fitness for cs in pop]
                new_pool = create_new_pool(
                    population=strings,
                    fitnesses=fitnesses,
                    generation=n,
                    translator=self.lsystem.hl_solver.translator,
                    n_individuals=2 * BIN_POP_SIZE,
                    minimize=minimize)
                axioms_sats = subdivide_axioms(hl_axioms=new_pool,
                                               lsystem=self.lsystem)
                for axiom in axioms_sats.keys():
                    cs = CandidateSolution(string=axiom)
                    if axioms_sats[axiom]['feasible']:
                        cs.is_feasible = True
                        cs.c_fitness = self.compute_fitness(
                            axiom=self.lsystem.hl_to_ll(axiom=axiom),
                            extra_args={
                                'alphabet':
                                    self.lsystem.ll_solver.atoms_alphabet
                            }) + (0.5 - axioms_sats[axiom]['n_constraints_v'])
                    else:
                        cs.c_fitness = axioms_sats[axiom]['n_constraints_v']
                    cs.set_content(
                        get_structure(
                            axiom=self.lsystem.hl_to_ll(axiom=axiom),
                            extra_args={
                                'alphabet':
                                    self.lsystem.ll_solver.atoms_alphabet
                            }))
                    self._set_behavior_descriptors(cs=cs)
                    cs.age = CS_MAX_AGE
                    generated.append(cs)
            self._update_bins(lcs=generated)
            self._check_res_trigger()
