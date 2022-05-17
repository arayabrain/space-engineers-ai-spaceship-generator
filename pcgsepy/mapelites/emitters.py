from abc import ABC, abstractmethod
from matplotlib.pyplot import grid
import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from pcgsepy.config import CS_MAX_AGE

from pcgsepy.mapelites.bin import MAPBin


class Emitter(ABC):
    def __init__(self) -> None:
        """Abstract class for Emitters.
        """
        super().__init__()
        self.name = 'abstract-emitter'
        self.requires_post = False
    
    @abstractmethod
    def pick_bin(self,
                 bins: np.ndarray) -> List[MAPBin]:
        """Abstract method for choosing a bin amongst multiple valid bins.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Raises:
            NotImplementedError: This method is abstract and must be overridden.

        Returns:
            MAPBin: The chosen bin.
        """
        raise NotImplementedError(f'The {self.name} must override the `pick_bin` method!')
    
    @abstractmethod
    def post_step(self):
        raise NotImplementedError(f'The {self.name} must override the `post_step` method!')


class RandomEmitter(Emitter):
    def __init__(self) -> None:
        """The random emitter class.
        """
        super().__init__()
        self.name = 'random-emitter'
    
    def pick_bin(self,
                 bins: np.ndarray) -> List[MAPBin]:
        """Randomly return a bin among possible valid bins.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Returns:
            MAPBin: The randomly picked bin.
        """
        bins = [b for b in bins.flatten().tolist() if len(b._feasible) > 0 or len(b._infeasible) > 0]
        fcs, ics = 0, 0
        selected = []
        while fcs < 2 and ics < 2:
            selected.append(bins.pop(np.random.choice(np.arange(len(bins)))))
            fcs += len(selected[-1]._feasible)
            ics += len(selected[-1]._infeasible)
        return selected
    
    def post_step(self):
        raise NotImplementedError('RandomEmitter has no `post_step`! Check `requires_post` before calling.')


class OptimisingEmitter(Emitter):
    def __init__(self) -> None:
        """The optimising emitter.
        """
        super().__init__()
        self.name = 'optimising-emitter'
    
    def pick_bin(self,
                 bins: np.ndarray) -> List[MAPBin]:
        """Select the bin whose elite content has the highest feasible fitness.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Returns:
            MAPBin: The selected bin.
        """
        bins = [b for b in bins.flatten().tolist() if len(b._feasible) > 0 or len(b._infeasible) > 0]
        sorted_bins = sorted(bins, key=lambda x: x.get_metric(metric='fitness', use_mean=True, population='feasible'), reverse=True)
        fcs, ics = 0, 0
        selected = []
        while fcs < 2 and ics < 2:
            selected.append(sorted_bins.pop(0))
            fcs += len(selected[-1]._feasible)
            ics += len(selected[-1]._infeasible)
        return selected
    
    def post_step(self):
        raise NotImplementedError('OptimisingEmitter has no `post_step`! Check `requires_post` before calling.')


class OptimisingEmitterV2(Emitter):
    def __init__(self) -> None:
        """The optimising emitter.
        """
        super().__init__()
        self.name = 'optimising-emitter-v2'
    
    def pick_bin(self,
                 bins: np.ndarray) -> List[List[MAPBin]]:
        """Select the bin whose elite content has the highest feasible fitness.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Returns:
            MAPBin: The selected bin.
        """
        bins = [b for b in bins.flatten().tolist() if len(b._feasible) > 0 or len(b._infeasible) > 0]
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
    
    def post_step(self):
        raise NotImplementedError('OptimisingEmitterV2 has no `post_step`! Check `requires_post` before calling.')


def get_emitter_by_str(emitter: str) -> Emitter:
    if emitter == 'random-emitter':
        return RandomEmitter()
    elif emitter == 'optimising-emitter':
        return OptimisingEmitter()
    elif emitter == 'optimising-emitter-v2':
        return OptimisingEmitterV2()
    else:
        raise NotImplementedError(f'Unrecognized emitter from string: {emitter}')
    

class HumanPrefMatrixEmitter(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'human-preference-matrix-emitter'
        self.requires_post = True
        self.tot_actions = 0
        self.decay = 1e-2
        self.last_selected = []
        self.prefs = None
    
    def build_pref_matrix(self,
                          bins: 'np.ndarray[MAPBin]') -> None:
        self.prefs = np.zeros(shape=bins.shape)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible') or bins[i, j].non_empty(pop='infeasbile'):
                    self.prefs[i, j] = 2 * self.decay
    
    def _random_bins(self,
                     bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        fcs, ics = 0, 0
        selected_bins = []
        idxs = np.argwhere(self.prefs > 0)
        np.random.shuffle(idxs)
        while fcs < 2 or ics < 2:
            self.last_selected.append(idxs[0, :])
            idxs = idxs[1:, :]
            b = bins[self.last_selected[-1][0], self.last_selected[-1][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
        return selected_bins
    
    def _most_likely_bins(self,
                          bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        fcs, ics = 0, 0
        selected_bins = []
        idxs = np.transpose(np.unravel_index(np.flip(np.argsort(self.prefs, axis=None)), self.prefs.shape))
        while fcs < 2 or ics < 2:
            self.last_selected.append(idxs[0, :])
            idxs = idxs[1:, :]
            b = bins[self.last_selected[-1][0], self.last_selected[-1][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
        return selected_bins
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self.prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self.last_selected = []
        p = np.random.uniform(low=0, high=1, size=1) < 1 / (1 + self.tot_actions)
        bins = self._random_bins(bins=bins) if p else self._most_likely_bins(bins=bins)
        self.tot_actions += 1
        return bins
    
    def _get_n_new_bins(self,
                        bins: 'np.ndarray[MAPBin]') -> int:
        n_new_bins = 0
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                b = bins[i, j]
                css = [*b._feasible, *b._infeasible]
                for cs in css:
                    if cs.age == CS_MAX_AGE:
                        n_new_bins += 1
                        break
        return n_new_bins
    
    def update_preferences(self,
                           idxs: List[Tuple[int, int]],
                           bins: 'np.ndarray[MAPBin]') -> None:
        assert self.prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        # get number of new/updated bins
        n_new_bins = self._get_n_new_bins(bins=bins)
        # update preference for selected bins
        for (i, j) in idxs:
            self.prefs[i, j] += 1.
            # if selected bin was just created, update parent bin accordingly
            if self.last_selected is not None:
                b = bins[i, j]
                css = [*b._feasible, *b._infeasible]
                for cs in css:
                    if cs.age == CS_MAX_AGE:
                        # we don't know which bin generated which, so update them all proportionally
                        for mn in self.last_selected:
                            self.prefs[mn[0], mn[1]] += 1 / n_new_bins
                        break
    
    def increase_preferences_res(self,
                                 idx: Tuple[int, int]) -> None:
        assert self.prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        i, j = idx
        # create new preference matrix by coping preferences over to new column/rows
        # rows repetitions
        a = np.ones(shape=self.prefs.shape[0], dtype=int)
        a[i] += 1
        # copy row
        self.prefs = np.repeat(self.prefs, repeats=a, axis=0)
        # columns repetitions
        a = np.ones(shape=self.prefs.shape[1], dtype=int)
        a[j] += 1
        # copy column
        self.prefs = np.repeat(self.prefs, repeats=a, axis=1)
    
    def _decay_preferences(self) -> None:
        assert self.prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self.prefs = self.prefs - self.decay
        self.prefs[np.where(self.prefs < 0)] = 0
    
    def post_step(self):
        assert self.prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self._decay_preferences()