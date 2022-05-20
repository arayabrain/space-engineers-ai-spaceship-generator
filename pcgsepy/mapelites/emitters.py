from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.pyplot import grid
from pcgsepy.config import CS_MAX_AGE
from pcgsepy.mapelites.bin import MAPBin
from pcgsepy.mapelites.buffer import Buffer, mean_merge
from sklearn.linear_model import LinearRegression, Ridge


class Emitter(ABC):
    def __init__(self) -> None:
        """Abstract class for Emitters.
        """
        super().__init__()
        self.name = 'abstract-emitter'
        self.requires_init = False
        self.requires_pre = False
        self.requires_post = False
    
    @abstractmethod
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        """Abstract method for choosing a bin amongst multiple valid bins.

        Args:
            bins (List[MAPBin]): The list of valid bins.

        Raises:
            NotImplementedError: This method is abstract and must be overridden.

        Returns:
            MAPBin: The chosen bin.
        """
        raise NotImplementedError(f'The {self.name} must override the `pick_bin` method!')
    
    def init_emitter(self,
                     **kwargs) -> None:
        raise NotImplementedError(f'The {self.name} must override the `init_emitter` method!')
    
    def pre_step(self,
                 **kwargs) -> None:
        raise NotImplementedError(f'The {self.name} must override the `pre_step` method!')
    
    def post_step(self,
                 **kwargs):
        raise NotImplementedError(f'The {self.name} must override the `post_step` method!')
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError(f'The {self.name} must override the `post_step` method!')


class RandomEmitter(Emitter):
    def __init__(self) -> None:
        """The random emitter class.
        """
        super().__init__()
        self.name = 'random-emitter'
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
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
    
    def reset(self) -> None:
        pass


class OptimisingEmitter(Emitter):
    def __init__(self) -> None:
        """The optimising emitter.
        """
        super().__init__()
        self.name = 'optimising-emitter'
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
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

    def reset(self) -> None:
        pass

class OptimisingEmitterV2(Emitter):
    def __init__(self) -> None:
        """The optimising emitter.
        """
        super().__init__()
        self.name = 'optimising-emitter-v2'
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[List[MAPBin]]:
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

    def reset(self) -> None:
        pass


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
    def __init__(self,
                 decay: float = 1e-2) -> None:
        super().__init__()
        self.name = 'human-preference-matrix-emitter'
        self.requires_init = True
        self.requires_post = True
        self.requires_pre = True
        
        self._tot_actions = 0
        self._decay = decay
        self._last_selected = []
        self._prefs = None
    
    def _build_pref_matrix(self,
                           bins: 'np.ndarray[MAPBin]') -> None:
        self._prefs = np.zeros(shape=bins.shape)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible') or bins[i, j].non_empty(pop='infeasbile'):
                    self._prefs[i, j] = 2 * self._decay
    
    def _random_bins(self,
                     bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        fcs, ics = 0, 0
        selected_bins = []
        idxs = np.argwhere(self._prefs > 0)
        np.random.shuffle(idxs)
        while fcs < 2 or ics < 2:
            self._last_selected.append(idxs[0, :])
            idxs = idxs[1:, :]
            b = bins[self._last_selected[-1][0], self._last_selected[-1][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
        return selected_bins
    
    def _most_likely_bins(self,
                          bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        fcs, ics = 0, 0
        selected_bins = []
        idxs = np.transpose(np.unravel_index(np.flip(np.argsort(self._prefs, axis=None)), self._prefs.shape))
        while fcs < 2 or ics < 2:
            self._last_selected.append(idxs[0, :])
            idxs = idxs[1:, :]
            b = bins[self._last_selected[-1][0], self._last_selected[-1][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
        return selected_bins
    
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
    
    def _increase_preferences_res(self,
                                  idx: Tuple[int, int]) -> None:
        assert self._prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        i, j = idx
        # create new preference matrix by coping preferences over to new column/rows
        # rows repetitions
        a = np.ones(shape=self._prefs.shape[0], dtype=int)
        a[i] += 1
        # copy row
        self._prefs = np.repeat(self._prefs, repeats=a, axis=0)
        # columns repetitions
        a = np.ones(shape=self._prefs.shape[1], dtype=int)
        a[j] += 1
        # copy column
        self._prefs = np.repeat(self._prefs, repeats=a, axis=1)
    
    def _decay_preferences(self) -> None:
        self._prefs -= self._prefs * self._decay
        self._prefs[np.where(self._prefs < 0)] = 0
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self._last_selected = []
        p = np.random.uniform(low=0, high=1, size=1) < 1 / (1 + self._tot_actions)
        bins = self._random_bins(bins=bins) if p else self._most_likely_bins(bins=bins)
        self._tot_actions += 1
        return bins
    
    def init_emitter(self,
                     **kwargs) -> None:
        assert self._prefs is None, f'{self.name} has already been initialized!'
        bins = kwargs['bins']
        self._build_pref_matrix(bins=bins)
        
    def pre_step(self, **kwargs) -> None:
        assert self._prefs is not None, f'{self.name} has not been initialized! Preference matrix has not been set.'
        bins = kwargs['bins']
        idxs = kwargs['selected_idxs']
        # get number of new/updated bins
        n_new_bins = self._get_n_new_bins(bins=bins)
        # update preference for selected bins
        for (i, j) in idxs:
            self._prefs[i, j] += 1.
            # if selected bin was just created, update parent bin accordingly
            if self._last_selected is not None:
                b = bins[i, j]
                css = [*b._feasible, *b._infeasible]
                for cs in css:
                    if cs.age == CS_MAX_AGE:
                        # we don't know which bin generated which, so update them all proportionally
                        for mn in self._last_selected:
                            self._prefs[mn[0], mn[1]] += 1 / n_new_bins
                        break
    
    def post_step(self,
                  bins: 'np.ndarray[MAPBin]') -> None:
        assert self._prefs is not None, f'{self.name} has not been initialized! Preference matrix has not been set.'
        self._decay_preferences()

    def reset(self) -> None:
        self._prefs = 0
        self._tot_actions = 0
        


class ContextualBanditEmitter(Emitter):
    def __init__(self,
                 epsilon: float = 0.2,
                 decay: float = 0.01,
                 n_features_context: int = 4) -> None:
        super().__init__()
        self.name = 'contextual-bandit-emitter'
        self.requires_pre = True
        
        self._initial_epsilon = epsilon
        self._epsilon = self._initial_epsilon
        self._decay = decay
        self._buffer = Buffer(merge_method=mean_merge)
        self._n_features_context = n_features_context
        self._estimator: LinearRegression = None
        self._fitted = False
        
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self._estimator = LinearRegression().fit(X=xs, y=ys)
        self._fitted = True
    
    def _extract_bin_context(self,
                             b: MAPBin) -> npt.NDArray[np.float32]:
        # TODO: Currently context is just feas. elite fitness metrics, but can be different
        return b.get_elite(population='feasible').fitness
    
    def _extract_context(self,
                         bins: 'np.ndarray[MAPBin]') -> npt.NDArray[np.float32]:
        bins = bins.flatten()
        context = np.zeros((bins.shape[0], self._n_features_context), dtype=np.float32)
        for i in range(bins.shape[0]):
            context[i, :] = self._extract_bin_context(bins[i])
        return context

    def _predict(self,
                 context: npt.NDArray[np.float32]) -> Tuple[int, int]:
        return np.flip(np.argsort(self._estimator.predict(X=context), axis=None))
    
    def pre_step(self, **kwargs) -> None:
        bins = kwargs['bins']
        idxs = kwargs['selected_idxs']
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                self._buffer.insert(x=self._extract_bin_context(bins[i, j]),
                                    y=1. if (i, j) in idxs else 0.)
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        context = self._extract_context(bins=bins)
        sorted_bins = np.transpose(np.unravel_index(self._predict(context=context), bins.shape))
        p = np.random.uniform(low=0, high=1, size=1) < self._epsilon
        self._epsilon -= self._epsilon * self._decay
        if p:
            np.random.shuffle(sorted_bins)
        fcs, ics, i = 0, 0, 0
        selected_bins = []
        while fcs < 2 or ics < 2:
            b = bins[sorted_bins[i][0], sorted_bins[i][1]]
            fcs += b.len('feasible')
            ics += b.len('infeasible')
            selected_bins.append(b)
            i += 1
        return selected_bins

    def reset(self) -> None:
        self._epsilon = self._initial_epsilon
        self._buffer.clear()
        self._estimator = None
        self._fitted = False