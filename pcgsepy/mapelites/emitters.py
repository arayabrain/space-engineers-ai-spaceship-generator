from abc import ABC, abstractmethod
import numpy as np
from typing import List

from pcgsepy.mapelites.bin import MAPBin


class Emitter(ABC):
    def __init__(self) -> None:
        """Abstract class for Emitters.
        """
        super().__init__()
        self.name = 'abstract-emitter'
    
    @abstractmethod
    def pick_bin(self,
                 bins: List[MAPBin]) -> MAPBin:
        """Abstract method for choosing a bin amongst multiple valid bins.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Raises:
            NotImplementedError: This method is abstract and must be overridden.

        Returns:
            MAPBin: The chosen bin.
        """
        raise NotImplementedError(f'The {self.name} must override the `pick_bin` method!')


class RandomEmitter(Emitter):
    def __init__(self) -> None:
        """The random emitter class.
        """
        super().__init__()
        self.name = 'random-emitter'
    
    def pick_bin(self,
                 bins: List[MAPBin]) -> MAPBin:
        """Randomly return a bin among possible valid bins.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Returns:
            MAPBin: The randomly picked bin.
        """
        fcs, ics = 0, 0
        selected = []
        while fcs < 2 and ics < 2:
            selected.append(bins.pop(np.random.choice(np.arange(len(bins)))))
            fcs += len(selected[-1]._feasible)
            ics += len(selected[-1]._infeasible)
        return selected


class OptimisingEmitter(Emitter):
    def __init__(self) -> None:
        """The optimising emitter.
        """
        super().__init__()
        self.name = 'optimising-emitter'
    
    def pick_bin(self,
                 bins: List[MAPBin]) -> List[MAPBin]:
        """Select the bin whose elite content has the highest feasible fitness.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Returns:
            MAPBin: The selected bin.
        """
        sorted_bins = sorted(bins, key=lambda x: x.get_metric(metric='fitness', use_mean=True, population='feasible'), reverse=True)
        fcs, ics = 0, 0
        selected = []
        while fcs < 2 and ics < 2:
            selected.append(sorted_bins.pop(0))
            fcs += len(selected[-1]._feasible)
            ics += len(selected[-1]._infeasible)
        return selected


class OptimisingEmitterV2(Emitter):
    def __init__(self) -> None:
        """The optimising emitter.
        """
        super().__init__()
        self.name = 'optimising-emitter-v2'
    
    def pick_bin(self,
                 bins: List[MAPBin]) -> List[List[MAPBin]]:
        """Select the bin whose elite content has the highest feasible fitness.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Returns:
            MAPBin: The selected bin.
        """
        sorted_bins_f = sorted(bins, key=lambda x: x.get_metric(metric='fitness', use_mean=True, population='feasible'), reverse=True)
        sorted_bins_i = sorted(bins, key=lambda x: x.get_metric(metric='fitness', use_mean=True, population='infeasible'), reverse=True)
        fcs, ics = 0, 0
        selected = [[], []]
        while fcs < 2:
            selected[0].append(sorted_bins_f.pop(0))
            fcs += len(selected[0][-1]._feasible)
        while ics < 2:
            selected[1].append(sorted_bins_i.pop(0))
            ics += len(selected[1][-1]._infeasible)
        return selected


def get_emitter_by_str(emitter: str) -> Emitter:
    if emitter == 'random-emitter':
        return RandomEmitter()
    elif emitter == 'optimising-emitter':
        return OptimisingEmitter()
    elif emitter == 'optimising-emitter-v2':
        return OptimisingEmitterV2()
    else:
        raise NotImplementedError(f'Unrecognized emitter from string: {emitter}')