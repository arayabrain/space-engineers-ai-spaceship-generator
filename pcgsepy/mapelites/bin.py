from typing import List, Tuple

import numpy as np

from ..config import BIN_POP_SIZE
from ..lsystem.solution import CandidateSolution


class MAPBin:

    def __init__(self,
                 bin_idx: Tuple[int, int],
                 bin_size: Tuple[float, float]):
        """Create a 2D bin object.

        Args:
            bin_idx (Tuple[int, int]): The index of the bin in the grid.
            bin_size (Tuple[float, float]): The size of the bin.
        """
        self._feasible = []
        self._infeasible = []
        self.bin_idx = bin_idx
        self.bin_size = bin_size

    def __str__(self) -> str:
        return f'Bin {self.bin_idx}, {self.bin_size} w/ {len(self._feasible)}f and {len(self._infeasible)}i cs'

    def __repr__(self) -> str:
        return str(self)

    def _reduce_pop(self,
                    pop: List[CandidateSolution]) -> List[CandidateSolution]:
        """Cull the population within this bin.

        Args:
            pop (List[CandidateSolution]): The population.

        Returns:
            List[CandidateSolution]: The culled population.
        """
        if len(pop) > BIN_POP_SIZE:
            pop.sort(key=lambda x: x.c_fitness, reverse=True)
            pop = pop[:BIN_POP_SIZE]
        return pop

    def insert_cs(self,
                  cs: CandidateSolution):
        """Add a solution in the bin.

        Args:
            cs (CandidateSolution): The solution to add.
        """
        if cs.is_feasible:
            if cs not in self._feasible:
                self._feasible.append(cs)
                self._feasible = self._reduce_pop(self._feasible)
        else:
            if cs not in self._infeasible:
                self._infeasible.append(cs)
                self._infeasible = self._reduce_pop(self._infeasible)

    def age(self,
            diff: int = -1):
        """Age the bin.

        Args:
            diff (int, optional): Value used to modify the bin's age. Defaults to -1.
        """
        for pop in [self._feasible, self._infeasible]:
            for cs in pop:
                cs.age += diff

    def remove_old(self):
        """Remove old solutions. Old solutions are solutions with an age `<=0`.
        """
        to_rem_f = [x for x in self._feasible if x.age <= 0]
        for cs in to_rem_f:
            self._feasible.remove(cs)
        to_rem_i = [x for x in self._infeasible if x.age <= 0]
        for cs in to_rem_i:
            self._infeasible.remove(cs)

    def get_metric(self,
                   metric: str,
                   use_mean: bool = True,
                   population: str = 'feasible') -> float:
        """Get the value for the given metric.

        Args:
            metric (str): The metric name.
            use_mean (bool, optional): Whether to compute the metric over the population or just the elite. Defaults to True.
            population (str, optional): Which population to compute the metric on. Defaults to 'feasible'.

        Raises:
            NotImplementedError: Exception raised if the metric is not recognized.

        Returns:
            float: The value of the metric.
        """
        op = np.mean if use_mean else np.max
        pop = self._feasible if population == 'feasible' else self._infeasible
        if metric == 'fitness':
            return op([cs.c_fitness for cs in pop]) if len(pop) > 0 else 0.
        elif metric == 'age':
            return op([cs.age for cs in pop]) if len(pop) > 0 else 0.
        elif metric == 'size':
            return len(pop)
        else:
            raise NotImplementedError(f'Unrecognized metric {metric}')

    def get_elite(self,
                  population: str = 'feasible') -> CandidateSolution:
        """Get the elite of the selected population.

        Args:
            population (str, optional): The population.. Defaults to 'feasible'.

        Returns:
            CandidateSolution: The elite solution.
        """
        pop = self._feasible if population == 'feasible' else self._infeasible
        get_max = True #if population == 'feasible' else False
        return sorted(pop, key=lambda x: x.c_fitness, reverse=get_max)[0]

    def toggle_module_mutability(self,
                                 module: str):
        """Toggle the mutability of a given module for all solutions in the bin.

        Args:
            module (str): The module name.
        """
        for pop in [self._feasible, self._infeasible]:
            for cs in pop:
                cs.hls_mod[module]['mutable'] = not cs.hls_mod[module]['mutable']
