from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pcgsepy.config import BIN_POP_SIZE, BIN_SMALLEST_PERC
from pcgsepy.lsystem.solution import CandidateSolution


class MAPBin:
    __slots__ = ['_feasible', '_infeasible', 'bin_idx', 'bin_size', 'bin_initial_size']

    def __init__(self,
                 bin_idx: Tuple[int, int],
                 bin_size: Tuple[float, float],
                 bin_initial_size: Optional[Tuple[float, float]]):
        """Create a 2D bin object.

        Args:
            bin_idx (Tuple[int, int]): The index of the bin in the grid.
            bin_size (Tuple[float, float]): The size of the bin.
        """
        self._feasible = []
        self._infeasible = []
        self.bin_idx = bin_idx
        self.bin_size = bin_size
        self.bin_initial_size = bin_initial_size if bin_initial_size else bin_size

    def __str__(self) -> str:
        return f'Bin {self.bin_idx}, {self.bin_size} w/ {len(self._feasible)}f and {len(self._infeasible)}i cs'

    def __repr__(self) -> str:
        return str(self)
    
    @property
    def subdividable(self) -> bool:
        """Check if the bin can be subdivided. Use this method when locally increasing resolution of the behavioral map.

        Returns:
            bool: Whether the bin can be subdivided.
        """
        bs0, bs1 = self.bin_size
        bis0, bis1 = self.bin_initial_size
        ms0 = bis0 * BIN_SMALLEST_PERC
        ms1 = bis1 * BIN_SMALLEST_PERC
        bs0 /= 2
        bs1 /= 2
        return bs0 >= ms0 and bs1 >= ms1
        

    def non_empty(self,
                  pop: str) -> bool:
        """Check if the bin is not empty for the given population.

        Args:
            pop (str): The population to check for.

        Returns:
            bool: Whether the bin is empty.
        """
        return len(self._feasible if pop == 'feasible' else self._infeasible) > 0

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
            diff (int, optional): Value used to modify the bin's age. Defaults to `-1`.
        """
        for pop in [self._feasible, self._infeasible]:
            for cs in pop:
                cs.age += diff

    def remove_old(self):
        """Remove old solutions. Old solutions are solutions with an age `<=0`."""
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
            use_mean (bool, optional): Whether to compute the metric over the population or just the elite. Defaults to `True`.
            population (str, optional): Which population to compute the metric on. Defaults to `'feasible'`.

        Raises:
            NotImplementedError: Raised if the metric is not recognized.

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
                  population: str = 'feasible',
                  always_max: bool = True) -> CandidateSolution:
        """Get the elite of the selected population.

        Args:
            population (str, optional): The population. Defaults to `'feasible'`.
            always_max (bool): Whether to select based on highest fitness. Defaults to `True`.

        Returns:
            CandidateSolution: The elite solution.
        """
        pop = self._feasible if population == 'feasible' else self._infeasible
        get_max = always_max or population == 'feasible'
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

    def to_json(self) -> Dict[str, Any]:
        return {
            'feasible': [cs.to_json() for cs in self._feasible],
            'infeasible': [cs.to_json() for cs in self._infeasible],
            'bin_idx': list(self.bin_idx),
            'bin_size': list(self.bin_size)
        }

    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'MAPBin':
        mb = MAPBin(bin_idx=tuple(my_args['bin_idx']),
                    bin_size=tuple(my_args['bin_size']))
        mb._feasible = [CandidateSolution.from_json(
            cs) for cs in my_args['feasible']]
        mb._infeasible = [CandidateSolution.from_json(
            cs) for cs in my_args['infeasible']]
        return mb
