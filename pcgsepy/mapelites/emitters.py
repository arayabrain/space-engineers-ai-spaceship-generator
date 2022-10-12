import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPRegressor
from pcgsepy.config import BETA_A, BETA_B, CONTEXT_IDXS, CS_MAX_AGE, USE_LINEAR_ESTIMATOR, USE_TORCH
from pcgsepy.mapelites.bin import MAPBin
from pcgsepy.mapelites.buffer import Buffer, mean_merge
from scipy.stats import boltzmann
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor

logging.getLogger('mapelites').info(msg=f'PyTorch set to {USE_TORCH}')

if USE_TORCH:
    from pcgsepy.nn.estimators import NonLinearEstimator, train_estimator
else:
    class NonLinearEstimator():
        def __init__(self,
                     xshape,
                     yshape):
            raise NotImplementedError('This object should never be instantiated')

        def train_estimator(estimator, xs, ys, n_epochs):
            raise NotImplementedError('This function should never be called')


def diversity_builder(bins: 'np.ndarray[MAPBin]',
                      n_features: int) -> npt.NDArray[np.float32]:
    
    def _distance(a: np.ndarray,
                  b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a - b)
    
    representations = np.zeros(shape=(bins.shape[0], bins.shape[1], n_features))
    for i in range(bins.shape[0]):
        for j in range(bins.shape[1]):
            if bins[i, j].non_empty(pop='feasible'):
                representations[i, j, :] = np.asarray(bins[i, j].get_elite(population='feasible').representation)
    representations[representations == 0] = np.nan
    mean_representation = np.nanmean(representations, axis=(0, 1))
    div = np.zeros(shape=bins.shape)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message='Mean of empty slice', category=RuntimeWarning)
        warnings.filterwarnings(action="ignore", message='All-NaN slice encountered', category=RuntimeWarning)
        for i in range(div.shape[0]):
            for j in range(div.shape[1]):
                div[i, j] = np.nanmean(_distance(representations[i, j],
                                                mean_representation))
        div = div / np.nanmax(div, axis=1)
        div[np.isnan(div)] = 0
    return div
    

class Emitter(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'abstract-emitter'
        self.requires_init = False
        self.requires_pre = False
        self.requires_post = False
        self.diversity_weight = 0.
    
    @abstractmethod
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
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
        raise NotImplementedError(f'The {self.name} must override the `reset` method!')


class RandomEmitter(Emitter):
    def __init__(self) -> None:
        """Create a random emitter class."""
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
        bins = [b for b in bins.flatten().tolist() if b.non_empty(pop='feasible') or b.non_empty(pop='infeasible')]
        fcs, ics = 0, 0
        selected = []
        while fcs < 2 or ics < 2:
            selected.append(bins.pop(np.random.choice(np.arange(len(bins)))))
            fcs += len(selected[-1]._feasible)
            ics += len(selected[-1]._infeasible)
        return selected
    
    def reset(self) -> None:
        pass

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'RandomEmitter':
        re = RandomEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        return re


class OptimisingEmitter(Emitter):
    def __init__(self) -> None:
        """Create an optimising emitter."""
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
        bins = [b for b in bins.flatten().tolist() if b.non_empty(pop='feasible') or b.non_empty(pop='infeasible')]
        sorted_bins = sorted(bins, key=lambda x: x.get_metric(metric='fitness', use_mean=True, population='feasible'), reverse=True)
        fcs, ics = 0, 0
        selected = []
        while fcs < 2 or ics < 2:
            selected.append(sorted_bins.pop(0))
            fcs += len(selected[-1]._feasible)
            ics += len(selected[-1]._infeasible)
        return selected

    def reset(self) -> None:
        pass
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'OptimisingEmitter':
        re = OptimisingEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        return re


class OptimisingEmitterV2(Emitter):
    def __init__(self) -> None:
        """Create an optimising emitter (population-based)."""
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
        bins = [b for b in bins.flatten().tolist() if b.non_empty(pop='feasible') or b.non_empty(pop='infeasible')]
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
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'OptimisingEmitterV2':
        re = OptimisingEmitterV2()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        return re


class GreedyEmitter(Emitter):
    def __init__(self) -> None:
        """Create a greedy emitter."""
        super().__init__()
        self.name = 'greedy-emitter'
        self.requires_pre = True
        self._last_selected: List[List[int]] = []
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        selected = [bins[idx] for idx in self._last_selected if bins[idx].non_empty(pop='feasible') or bins[idx].non_empty(pop='infeasbile')]
        return selected
    
    def reset(self) -> None:
        self._last_selected = []

    def pre_step(self, **kwargs) -> None:
        self._last_selected = []
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        for idx in idxs:
            self._last_selected.append(idx)
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            'last_selected': self._last_selected
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'GreedyEmitter':
        re = GreedyEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        re._last_selected = my_args['last_selected']
        return re


class HumanPrefMatrixEmitter(Emitter):
    def __init__(self,
                 decay: float = 1e-2) -> None:
        """Create a human preference-matrix emitter.

        Args:
            decay (float, optional): The preference decay. Defaults to `1e-2`.
        """
        super().__init__()
        self.name = 'human-preference-matrix-emitter'
        self.requires_init = True
        self.requires_post = True
        self.requires_pre = True
        
        self._tot_actions = 0
        self._decay = decay
        self._last_selected = []
        self._prefs = None
        
        self.sampling_strategy = 'epsilon_greedy'  # or 'gibbs', 'thompson'
        self._thompson_stats = None
    
    def _build_pref_matrix(self,
                           bins: 'np.ndarray[MAPBin]') -> None:
        """Build the preference matrix.

        Args:
            bins (np.ndarray[MAPBin]): The MAP-Elites bins.
        """
        self._prefs = np.zeros(shape=bins.shape)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible') or bins[i, j].non_empty(pop='infeasbile'):
                    self._prefs[i, j] = 2 * self._decay
        self._thompson_stats = np.ones(shape=(self._prefs.shape[0], self._prefs.shape[1], 2))
        self._thompson_stats[:,:,0] = BETA_A * self._thompson_stats[:,:,0]
        self._thompson_stats[:,:,1] = BETA_B * self._thompson_stats[:,:,1]
    
    def _random_bins(self,
                     bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        fcs, ics = 0, 0
        selected_bins = []
        choose_from = self._prefs * (1 - self.diversity_weight) #+ diversity_builder(bins=bins, n_features=7) * self.diversity_weight
        idxs = np.argwhere(choose_from > 0)
        np.random.shuffle(idxs)
        while fcs < 1 or ics < 1:
            self._last_selected.append(idxs[0, :])
            idxs = idxs[1:, :]
            b = bins[self._last_selected[-1][0], self._last_selected[-1][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
        return selected_bins
    
    def _most_likely_bins(self,
                          bins: 'np.ndarray[MAPBin]',
                          prefs: Optional[np.typing.NDArray] = None) -> List[MAPBin]:
        fcs, ics = 0, 0
        selected_bins = []
        prefs = prefs if prefs is not None else self._prefs
        choose_from = prefs * (1 - self.diversity_weight) #+ diversity_builder(bins=bins, n_features=7) * self.diversity_weight
        idxs = np.transpose(np.unravel_index(np.flip(np.argsort(choose_from, axis=None)), prefs.shape))
        while fcs < 1 or ics < 1:
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
    
    def _reshape_matrix(self,
                        arr: np.typing.NDArray,
                        idx: Tuple[int, int]) -> np.typing.NDArray:
        i, j = idx
        # create new matrix by coping preferences over to new column/rows
        # rows repetitions
        a = np.ones(shape=arr.shape[0], dtype=int)
        a[i] += 1
        # copy row
        arr = np.repeat(arr, repeats=a, axis=0)
        # columns repetitions
        a = np.ones(shape=arr.shape[1], dtype=int)
        a[j] += 1
        # copy column
        arr = np.repeat(arr, repeats=a, axis=1)
        return arr
    
    def _increase_preferences_res(self,
                                  idx: Tuple[int, int]) -> None:
        assert self._prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self._prefs = self._reshape_matrix(arr=self._prefs,
                                           idx=idx)
        self._thompson_stats = self._reshape_matrix(arr=self._thompson_stats,
                                                    idx=idx)
    
    def _decay_preferences(self) -> None:
        self._prefs -= self._prefs * self._decay
        self._prefs[np.where(self._prefs < 0)] = 0
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self._last_selected = []
        if self.sampling_strategy in ['epsilon_greedy', 'gibbs']:
            if self.sampling_strategy == 'epsilon_greedy':
                p = np.random.uniform(low=0, high=1, size=1) < 1 / (1 + self._tot_actions)
            elif self.sampling_strategy == 'gibbs':
                p = np.random.uniform(low=0, high=1, size=1) < boltzmann.ppf(self._tot_actions, 0.5, 1)
            selected_bins = self._random_bins(bins=bins) if p else self._most_likely_bins(bins=bins)
        elif self.sampling_strategy == 'thompson':
            prob_matrix = np.zeros_like(self._prefs)
            for i in range(prob_matrix.shape[0]):
                for j in range(prob_matrix.shape[1]):
                    prob_matrix[i, j] = np.random.beta(a=self._thompson_stats[i, j, 0],
                                                       b=self._thompson_stats[i, j, 1],
                                                       size=1)
            scaled_prefs = self._prefs * prob_matrix
            selected_bins = self._most_likely_bins(bins=bins,
                                                   prefs=scaled_prefs)
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        self._tot_actions += 1
        return selected_bins
    
    def init_emitter(self,
                     **kwargs) -> None:
        assert self._prefs is None, f'{self.name} has already been initialized!'
        bins = kwargs['bins']
        self._build_pref_matrix(bins=bins)
        
    def pre_step(self, **kwargs) -> None:
        assert self._prefs is not None, f'{self.name} has not been initialized! Preference matrix has not been set.'
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
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
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if [i, j] in idxs:
                    self._thompson_stats[i, j, 0] += 1
                else:
                    self._thompson_stats[i, j, 1] += 1
    
    def post_step(self,
                  bins: 'np.ndarray[MAPBin]') -> None:
        assert self._prefs is not None, f'{self.name} has not been initialized! Preference matrix has not been set.'
        self._decay_preferences()

    def reset(self) -> None:
        self._prefs = None
        self._thompson_stats = None
        self._tot_actions = 0
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            'tot_actions': self._tot_actions,
            'decay': self._decay,
            'last_selected': self._last_selected,  # may need conversion tolist()
            'prefs': self._prefs.tolist(),
            'thompson_stats': self._thompson_stats.tolist()
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'HumanPrefMatrixEmitter':
        re = HumanPrefMatrixEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        re._tot_actions = my_args['tot_actions']
        re._decay = my_args['decay']
        re._last_selected = my_args['last_selected']  # may need conversion np.asarray
        re._prefs = np.asarray(my_args['prefs'])
        re._thompson_stats = np.asarray(my_args['thompson_stats'])
        return re


class ContextualBanditEmitter(Emitter):
    def __init__(self,
                 epsilon: float = 0.2,
                 decay: float = 0.01,
                 n_features_context: int = len(CONTEXT_IDXS)) -> None:
        super().__init__()
        self.name = 'contextual-bandit-emitter'
        self.requires_pre = True
        
        self._initial_epsilon = epsilon
        self._epsilon = self._initial_epsilon
        self._decay = decay
        self._buffer = Buffer(merge_method=mean_merge)
        self._n_features_context = n_features_context
        self._estimator: Union[LogisticRegression, NonLinearEstimator, MLPRegressor] = None
        self._fitted = False
        
        self._tot_actions = 0
        self.sampling_strategy = 'thompson'  # or 'gibbs', 'epsilon_greedy'
        self._thompson_stats = {}
    
    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        if USE_LINEAR_ESTIMATOR:
            self._estimator = LogisticRegression().fit(X=xs, y=ys)
        elif USE_TORCH:
            self._estimator = NonLinearEstimator(xshape=self._n_features_context,
                                                 yshape=1)
            train_estimator(estimator=self._estimator,
                            xs=xs,
                            ys=ys,
                            n_epochs=20)
        else:
            self._estimator = MLPRegressor(hidden_layer_sizes=self._n_features_context,
                                           activation='relu',
                                           solver='sgd',
                                           max_iter=20).fit(X=xs, y=ys)
        self._fitted = True
    
    def _extract_bin_context(self,
                             b: MAPBin) -> npt.NDArray[np.float32]:
        return np.asarray(b.get_elite(population='feasible').representation)[CONTEXT_IDXS]
    
    def _extract_context(self,
                         bins: 'np.ndarray[MAPBin]') -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        bins = bins.flatten()
        context = np.zeros((bins.shape[0], self._n_features_context), dtype=np.float32)
        mask = np.zeros((bins.shape[0]), dtype=np.uint8)
        for i in range(bins.shape[0]):
            if bins[i].non_empty(pop='feasible'):
                context[i, :] = self._extract_bin_context(bins[i])
                mask[i] = 1
        return context, mask

    def _predict(self,
                 bins: 'np.ndarray[MAPBin]') -> Tuple[int, int]:
        context, mask = self._extract_context(bins=bins)
        choose_from = self._estimator.predict(X=context) #* (1 - self.diversity_weight) + diversity_builder(bins=bins, n_features=self._n_features_context).flatten() * self.diversity_weight
        if self.sampling_strategy == 'thompson':
            for i in range(context.shape[0]):
                # if bins[i // bins.shape[0], i % bins.shape[1]].non_empty(pop='feasible'):
                c = context[i].tobytes()
                scale_factor = np.random.beta(a=BETA_A + (self._thompson_stats[c][0] if c in self._thompson_stats else 1),
                                              b=BETA_B + (self._thompson_stats[c][1] if c in self._thompson_stats else 1 + self._tot_actions),
                                              size=1)
                choose_from[i] *= scale_factor
        choose_from *= mask
        return np.flip(np.argsort(choose_from, axis=None))
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    self._buffer.insert(x=self._extract_bin_context(bins[i, j]),
                                        y=1. if (i, j) in idxs else 0.)
                    context = self._extract_bin_context(b=bins[i, j]).tobytes()
                    if (i, j) in idxs:
                        if context not in self._thompson_stats:
                            self._thompson_stats[context] = [2, 1]
                        else:
                            self._thompson_stats[context][0] += 1
                    else:
                        if context not in self._thompson_stats:
                            self._thompson_stats[context] = [1, 2]
                        else:
                            self._thompson_stats[context][1] += 1
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        sorted_bins = np.transpose(np.unravel_index(self._predict(bins=bins), bins.shape))
        if self.sampling_strategy == 'epsilon_greedy':
            p = np.random.uniform(low=0, high=1, size=1) < self._epsilon
            self._epsilon -= self._epsilon * self._decay
        elif self.sampling_strategy == 'gibbs':
            lambda_, N = 1.4, self._tot_actions
            p = np.random.uniform(low=0, high=1, size=1) < boltzmann.pmf(self._tot_actions, lambda_, N)
            self._tot_actions += 1
        elif self.sampling_strategy == 'thompson':
            p = None
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        if p:
            np.random.shuffle(sorted_bins)
        fcs, ics, i = 0, 0, 0
        selected_bins = []
        while fcs < 1 or ics < 1:
            b = bins[sorted_bins[i][0], sorted_bins[i][1]]
            df, di = len(b._feasible), len(b._infeasible)
            if df > 0 or di > 0:
                fcs += df
                ics += di
                selected_bins.append(b)
            i += 1
        return selected_bins

    def reset(self) -> None:
        self._epsilon = self._initial_epsilon
        self._buffer.clear()
        self._thompson_stats = {}
        self._estimator = None
        self._fitted = False
    
    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self._epsilon,
            'decay': self._decay,
            'buffer': self._buffer.to_json(),
            'n_features_context': self._n_features_context,
            'diversity_weight': self.diversity_weight,
            'fitted': self._fitted,
            
            'thompson_stats': self._thompson_stats
        }
        j['estimator_name'] = self._estimator.__class__.__name__ if self._estimator else None
        if isinstance(self._estimator, LogisticRegression):
            j['estimator_params'] = self._estimator.get_params(),
            j['estimator_coefs'] = self._estimator.coef_.tolist() if self._fitted else None,
            j['estimator_intercept'] = np.asarray(self._estimator.intercept_).tolist() if self._fitted else None,
        elif isinstance(self._estimator, MLPRegressor):
            j['coefs_'] = self._estimator.coefs_
            j['intercepts_']: self._estimator.intercepts_
            j['n_features_in_']: self._estimator.n_features_in_
            j['n_iter_']: self._estimator.n_iter_
            j['n_layers_']: self._estimator.n_layers_
            j['n_outputs_']: self._estimator.n_outputs_
            j['out_activation_']: self._estimator.out_activation_
        elif USE_TORCH and isinstance(self._estimator, NonLinearEstimator):
            j['estimator_parameters'] = self._estimator.to_json()
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'ContextualBanditEmitter':
        re = ContextualBanditEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._epsilon = my_args['epsilon']
        re._decay = my_args['decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._n_features_context = my_args['n_features_context']
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
            if my_args['estimator_name'] == 'NonLinearEstimator' and USE_TORCH and not USE_LINEAR_ESTIMATOR:
                re._estimator = NonLinearEstimator.from_json(my_args=my_args['estimator_parameters'])
            elif my_args['estimator_name'] == 'LogisticRegression' and USE_LINEAR_ESTIMATOR:
                re._estimator = LogisticRegression()
                re._estimator.set_params(my_args['estimator_params'])
                if my_args['estimator_coefs'] is not None:
                    re._estimator.coef_ = np.asarray(my_args['estimator_coefs'])
                if my_args['estimator_intercept'] is not None:
                    re._estimator.intercept_ = np.asarray(my_args['estimator_intercept'])
            elif my_args['estimator_name'] == 'MLPRegressor':
                re._estimator = MLPRegressor()
                re._estimator.coefs_ = my_args['coefs_']
                re._estimator.intercepts_ = my_args['intercepts_']
                re._estimator.n_features_in_ = my_args['n_features_in_']
                re._estimator.n_iter_ = my_args['n_iter_']
                re._estimator.n_layers_ = my_args['n_layers_']
                re._estimator.n_outputs_ = my_args['n_outputs_']
                re._estimator.out_activation_ = my_args['out_activation_']
            else:
                raise ValueError(f'Unrecognized estimator name: {my_args["estimator_name"]}.')
        
            
        
        re._thompson_stats = my_args['thompson_stats']
        
        return re


class PreferenceBanditEmitter(Emitter):
    def __init__(self,
                 epsilon: float = 0.2,
                 decay: float = 0.01) -> None:
        super().__init__()
        self.name = 'preference-bandit-emitter'
        self.requires_pre = True
        
        self._initial_epsilon = epsilon
        self._epsilon = self._initial_epsilon
        self._decay = decay
        self._buffer = Buffer(merge_method=mean_merge)
        self._estimator: Union[LogisticRegression, NonLinearEstimator, MLPRegressor] = None
        self._fitted = False
        
        self._tot_actions = 0
        self.sampling_strategy = 'epsilon_greedy'  # or 'gibbs', 'thompson'
        self._thompson_stats = {}

    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        if USE_LINEAR_ESTIMATOR:
            self._estimator = LogisticRegression().fit(X=xs, y=ys)
        elif USE_TORCH:
            self._estimator = NonLinearEstimator(xshape=2,
                                                 yshape=1)
            train_estimator(estimator=self._estimator,
                            xs=xs,
                            ys=ys,
                            n_epochs=20)
        else:
            self._estimator = MLPRegressor(hidden_layer_sizes=2,
                                           activation='relu',
                                           solver='sgd',
                                           max_iter=20).fit(X=xs, y=ys)
        self._fitted = True
    
    def _get_valid_bins(self,
                        bins: 'np.ndarray[MAPBin]') -> npt.NDArray[np.float32]:
        valid = np.zeros_like(bins, dtype=np.uint8)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    valid[i, j] = 1
        return valid
    
    def _get_bins_index(self,
                        bins: 'np.ndarray[MAPBin]',
                        normalize: bool = True) -> npt.NDArray[np.float32]:
        bin_idxs = np.zeros(shape=(bins.shape[0], bins.shape[1], 2))
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                bin_idxs[i, j, :] = [i, j]
        if normalize:
            bin_idxs[:, :, 0] = bin_idxs[:, :, 0] / bins.shape[0]
            bin_idxs[:, :, 1] = bin_idxs[:, :, 1] / bins.shape[1]
        return bin_idxs
    
    def _predict(self,
                 bins: 'np.ndarray[MAPBin]') -> Tuple[int, int]:
        mask = self._get_valid_bins(bins=bins)
        preferences = self._get_bins_index(bins=bins, normalize=True)
        mask3d = np.zeros(preferences.shape, dtype=bool)
        mask3d[:,:,:] = mask[:,:, np.newaxis] == 1
        preferences = preferences[mask3d].reshape(-1, 2)
        choose_from = self._estimator.predict(X=preferences)
        out = np.zeros_like(bins, dtype=np.float32)
        out[mask == 1] = choose_from[:]
        if self.sampling_strategy == 'thompson':
            for i in range(bins.shape[0]):
                for j in range(bins.shape[1]):
                    n_i = i / bins.shape[0]
                    n_j = j / bins.shape[1]
                    scale_factor = np.random.beta(a=BETA_A + (self._thompson_stats[n_i, n_j][0] if (n_i, n_j) in self._thompson_stats else 1),
                                                  b=BETA_B + (self._thompson_stats[n_i, n_j][1] if (n_i, n_j) in self._thompson_stats else 1 + self._tot_actions),
                                                  size=1)
                    out[i, j] *= scale_factor        
        return np.flip(np.argsort(out, axis=None))
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        bounds: List[Tuple[float, float]] = kwargs['bounds']
        bcs0 = np.cumsum([bounds[0][0]] + [b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([bounds[1][0]] + [b.bin_size[1] for b in bins[:, 0]])[:-1]
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    self._buffer.insert(x=np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]]),
                                        y=1. if (i, j) in idxs else 0.)
            n_i = i / bins.shape[0]
            n_j = j / bins.shape[1]
            if (i, j) in idxs:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [2, 1]
                else:
                    self._thompson_stats[n_i, n_j][0] += 1
            else:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [1, 2]
                else:
                    self._thompson_stats[n_i, n_j][1] += 1
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        sorted_bins = np.transpose(np.unravel_index(self._predict(bins=bins), bins.shape))
        if self.sampling_strategy == 'epsilon_greedy':
            p = np.random.uniform(low=0, high=1, size=1) < self._epsilon
            self._epsilon -= self._epsilon * self._decay
        elif self.sampling_strategy == 'gibbs':
            lambda_, N = 1.4, self._tot_actions
            p = np.random.uniform(low=0, high=1, size=1) < boltzmann.rvs(lambda_, N, loc=0, size=1)
            self._tot_actions += 1
        elif self.sampling_strategy == 'thompson':
            p = None
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        if p:
            np.random.shuffle(sorted_bins)
        fcs, ics, i = 0, 0, 0
        selected_bins = []
        while fcs < 1 or ics < 1:
            b = bins[sorted_bins[i][0], sorted_bins[i][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            i += 1
        return selected_bins

    def reset(self) -> None:
        self._epsilon = self._initial_epsilon
        self._buffer.clear()
        self._thompson_stats = {}
        self._estimator = None
        self._fitted = False
        self._tot_actions = 0

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            'tot_actions': self._tot_actions,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self._epsilon,
            'decay': self._decay,
            'buffer': self._buffer.to_json(),
            'diversity_weight': self.diversity_weight,
            'fitted': self._fitted,
            
            'thompson_stats': self._thompson_stats
        }
        j['estimator_name'] = self._estimator.__class__.__name__ if self._estimator else None
        if isinstance(self._estimator, LogisticRegression):
            j['estimator_params'] = self._estimator.get_params(),
            j['estimator_coefs'] = self._estimator.coef_.tolist() if self._fitted else None,
            j['estimator_intercept'] = np.asarray(self._estimator.intercept_).tolist() if self._fitted else None,
        elif isinstance(self._estimator, MLPRegressor):
            j['coefs_'] = self._estimator.coefs_
            j['intercepts_']: self._estimator.intercepts_
            j['n_features_in_']: self._estimator.n_features_in_
            j['n_iter_']: self._estimator.n_iter_
            j['n_layers_']: self._estimator.n_layers_
            j['n_outputs_']: self._estimator.n_outputs_
            j['out_activation_']: self._estimator.out_activation_
        elif USE_TORCH and isinstance(self._estimator, NonLinearEstimator):
            j['estimator_parameters'] = self._estimator.to_json()
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'PreferenceBanditEmitter':
        re = PreferenceBanditEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        re._tot_actions = my_args['tot_actions']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._epsilon = my_args['epsilon']
        re._decay = my_args['decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._n_features_context = my_args['n_features_context']
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
            if my_args['estimator_name'] == 'NonLinearEstimator' and USE_TORCH and not USE_LINEAR_ESTIMATOR:
                re._estimator = NonLinearEstimator.from_json(my_args=my_args['estimator_parameters'])
            elif my_args['estimator_name'] == 'LogisticRegression' and USE_LINEAR_ESTIMATOR:
                re._estimator = LogisticRegression()
                re._estimator.set_params(my_args['estimator_params'])
                if my_args['estimator_coefs'] is not None:
                    re._estimator.coef_ = np.asarray(my_args['estimator_coefs'])
                if my_args['estimator_intercept'] is not None:
                    re._estimator.intercept_ = np.asarray(my_args['estimator_intercept'])
            elif my_args['estimator_name'] == 'MLPRegressor':
                re._estimator = MLPRegressor()
                re._estimator.coefs_ = my_args['coefs_']
                re._estimator.intercepts_ = my_args['intercepts_']
                re._estimator.n_features_in_ = my_args['n_features_in_']
                re._estimator.n_iter_ = my_args['n_iter_']
                re._estimator.n_layers_ = my_args['n_layers_']
                re._estimator.n_outputs_ = my_args['n_outputs_']
                re._estimator.out_activation_ = my_args['out_activation_']
            else:
                raise ValueError(f'Unrecognized estimator name: {my_args["estimator_name"]}.')
        
        re._thompson_stats = my_args['thompson_stats']
        
        return re


class KNNEmitter(Emitter):
    def __init__(self,
                 epsilon: float = 0.2,
                 decay: float = 0.01) -> None:
        super().__init__()
        self.name = 'knn-emitter'
        self.requires_pre = True
        
        self._initial_epsilon = epsilon
        self._epsilon = self._initial_epsilon
        self._decay = decay
        self._buffer = Buffer(merge_method=mean_merge)
        self._estimator: KNeighborsRegressor = None
        self._fitted = False
        
        self._tot_actions = 0
        self.sampling_strategy = 'thompson'  # or 'gibbs', 'thompson'
        self._thompson_stats = {}

    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self._estimator = KNeighborsRegressor().fit(X=xs, y=ys)
        logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self._estimator.score(xs, ys):.2%}')
        self._fitted = True
    
    def _get_valid_bins(self,
                        bins: 'np.ndarray[MAPBin]') -> npt.NDArray[np.float32]:
        valid = np.zeros_like(bins, dtype=np.uint8)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    valid[i, j] = 1
        return valid
    
    def _get_bins_index(self,
                        bins: 'np.ndarray[MAPBin]',
                        normalize: bool = True) -> npt.NDArray[np.float32]:
        bin_idxs = np.zeros(shape=(bins.shape[0], bins.shape[1], 2))
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                bin_idxs[i, j, :] = [i, j]
        if normalize:
            bin_idxs[:, :, 0] = bin_idxs[:, :, 0] / bins.shape[0]
            bin_idxs[:, :, 1] = bin_idxs[:, :, 1] / bins.shape[1]
        return bin_idxs
    
    def _predict(self,
                 bins: 'np.ndarray[MAPBin]') -> Tuple[int, int]:
        mask = self._get_valid_bins(bins=bins)
        preferences = self._get_bins_index(bins=bins, normalize=True)
        mask3d = np.zeros(preferences.shape, dtype=bool)
        mask3d[:,:,:] = mask[:,:, np.newaxis] == 1
        predictions = self._estimator.predict(X=preferences[mask3d].reshape(-1, 2))
        logging.getLogger('mapelites').debug(f'[{__name__}._predict] {predictions=}')
        out = np.zeros_like(bins, dtype=np.float32)
        ii, jj = np.nonzero(mask)
        for i, j, v in zip(ii, jj, predictions):
            out[i, j] = v
        if self.sampling_strategy == 'thompson':
            for i, j in zip(ii, jj):
                n_i = i / bins.shape[0]
                n_j = j / bins.shape[1]
                scale_factor = np.random.beta(a=BETA_A + (self._thompson_stats[n_i, n_j][0] if (n_i, n_j) in self._thompson_stats else 1),
                                                b=BETA_B + (self._thompson_stats[n_i, n_j][1] if (n_i, n_j) in self._thompson_stats else 1 + self._tot_actions),
                                                size=1)
                out[i, j] *= scale_factor
        out = out.flatten()
        order = np.flip(np.argsort(out, axis=None))
        out = out[order]
        out_idxs = np.arange(len(out))[order][out != 0.]
        logging.getLogger('mapelites').debug(f'[{__name__}._predict] sorted predicted indices={out_idxs}; predicted values={out[out != 0.]}')
        return out_idxs
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        bounds: List[Tuple[float, float]] = kwargs['bounds']
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        bcs0 = np.cumsum([bounds[0][0]] + [b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([bounds[1][0]] + [b.bin_size[1] for b in bins[:, 0]])[:-1]
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {(i, j)=}; x={np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]])}; dy={1. if (i, j) in idxs else 0.}')
                    self._buffer.insert(x=np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]]),
                                        y=1. if (i, j) in idxs else 0.)
            n_i = i / bins.shape[0]
            n_j = j / bins.shape[1]
            if (i, j) in idxs:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [2, 1]
                else:
                    self._thompson_stats[n_i, n_j][0] += 1
            else:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [1, 2]
                else:
                    self._thompson_stats[n_i, n_j][1] += 1
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        sorted_bins = np.transpose(np.unravel_index(self._predict(bins=bins), bins.shape))
        logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {sorted_bins=}')
        if self.sampling_strategy == 'epsilon_greedy':
            p = np.random.uniform(low=0, high=1, size=1) < self._epsilon
            self._epsilon -= self._epsilon * self._decay
        elif self.sampling_strategy == 'gibbs':
            lambda_, N = 1.4, self._tot_actions
            p = np.random.uniform(low=0, high=1, size=1) < boltzmann.rvs(lambda_, N, loc=0, size=1)
            self._tot_actions += 1
        elif self.sampling_strategy == 'thompson':
            p = None
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        if p:
            np.random.shuffle(sorted_bins)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}; {sorted_bins=}')
        fcs, ics, i = 0, 0, 0
        selected_bins = []
        while fcs < 1 or ics < 1:
            b = bins[sorted_bins[i][0], sorted_bins[i][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            i += 1
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {fcs=}; {ics=}; {i=}')
        return selected_bins

    def reset(self) -> None:
        self._epsilon = self._initial_epsilon
        self._buffer.clear()
        self._thompson_stats = {}
        self._estimator = None
        self._fitted = False
        self._tot_actions = 0

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            'tot_actions': self._tot_actions,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self._epsilon,
            'decay': self._decay,
            'buffer': self._buffer.to_json(),
            'diversity_weight': self.diversity_weight,
            'fitted': self._fitted,
            
            'thompson_stats': self._thompson_stats
        }
        j['estimator_name'] = self._estimator.__class__.__name__ if self._estimator else None
        if isinstance(self._estimator, KNeighborsRegressor):
            pass
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'KNNEmitter':
        re = KNNEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        re._tot_actions = my_args['tot_actions']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._epsilon = my_args['epsilon']
        re._decay = my_args['decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._n_features_context = my_args['n_features_context']
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
                re._estimator = KNeighborsRegressor()
                # TODO: load parameters
        
        re._thompson_stats = my_args['thompson_stats']
        
        return re


class LinearKernelEmitter(Emitter):
    def __init__(self,
                 epsilon: float = 0.2,
                 decay: float = 0.01) -> None:
        super().__init__()
        self.name = 'linear-kernel-emitter'
        self.requires_pre = True
        
        self._initial_epsilon = epsilon
        self._epsilon = self._initial_epsilon
        self._decay = decay
        self._buffer = Buffer(merge_method=mean_merge)
        self._estimator: KNeighborsRegressor = None
        self._fitted = False
        
        self._tot_actions = 0
        self.sampling_strategy = 'thompson'  # or 'gibbs', 'thompson'
        self._thompson_stats = {}

    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self._estimator = KernelRidge(kernel='linear').fit(X=xs, y=ys)
        logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self._estimator.score(xs, ys):.2%}')
        self._fitted = True
    
    def _get_valid_bins(self,
                        bins: 'np.ndarray[MAPBin]') -> npt.NDArray[np.float32]:
        valid = np.zeros_like(bins, dtype=np.uint8)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    valid[i, j] = 1
        return valid
    
    def _get_bins_index(self,
                        bins: 'np.ndarray[MAPBin]',
                        normalize: bool = True) -> npt.NDArray[np.float32]:
        bin_idxs = np.zeros(shape=(bins.shape[0], bins.shape[1], 2))
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                bin_idxs[i, j, :] = [i, j]
        if normalize:
            bin_idxs[:, :, 0] = bin_idxs[:, :, 0] / bins.shape[0]
            bin_idxs[:, :, 1] = bin_idxs[:, :, 1] / bins.shape[1]
        return bin_idxs
    
    def _predict(self,
                 bins: 'np.ndarray[MAPBin]') -> Tuple[int, int]:
        mask = self._get_valid_bins(bins=bins)
        preferences = self._get_bins_index(bins=bins, normalize=True)
        mask3d = np.zeros(preferences.shape, dtype=bool)
        mask3d[:,:,:] = mask[:,:, np.newaxis] == 1
        predictions = self._estimator.predict(X=preferences[mask3d].reshape(-1, 2))
        logging.getLogger('mapelites').debug(f'[{__name__}._predict] {predictions=}')
        out = np.zeros_like(bins, dtype=np.float32)
        ii, jj = np.nonzero(mask)
        for i, j, v in zip(ii, jj, predictions):
            out[i, j] = v
        if self.sampling_strategy == 'thompson':
            for i, j in zip(ii, jj):
                n_i = i / bins.shape[0]
                n_j = j / bins.shape[1]
                scale_factor = np.random.beta(a=BETA_A + (self._thompson_stats[n_i, n_j][0] if (n_i, n_j) in self._thompson_stats else 1),
                                                b=BETA_B + (self._thompson_stats[n_i, n_j][1] if (n_i, n_j) in self._thompson_stats else 1 + self._tot_actions),
                                                size=1)
                out[i, j] *= scale_factor
        out = out.flatten()
        order = np.flip(np.argsort(out, axis=None))
        out = out[order]
        out_idxs = np.arange(len(out))[order][out != 0.]
        logging.getLogger('mapelites').debug(f'[{__name__}._predict] sorted predicted indices={out_idxs}; predicted values={out[out != 0.]}')
        return out_idxs
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        bounds: List[Tuple[float, float]] = kwargs['bounds']
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        bcs0 = np.cumsum([bounds[0][0]] + [b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([bounds[1][0]] + [b.bin_size[1] for b in bins[:, 0]])[:-1]
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {(i, j)=}; x={np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]])}; dy={1. if (i, j) in idxs else 0.}')
                    self._buffer.insert(x=np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]]),
                                        y=1. if (i, j) in idxs else 0.)
            n_i = i / bins.shape[0]
            n_j = j / bins.shape[1]
            if (i, j) in idxs:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [2, 1]
                else:
                    self._thompson_stats[n_i, n_j][0] += 1
            else:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [1, 2]
                else:
                    self._thompson_stats[n_i, n_j][1] += 1
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        sorted_bins = np.transpose(np.unravel_index(self._predict(bins=bins), bins.shape))
        logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {sorted_bins=}')
        if self.sampling_strategy == 'epsilon_greedy':
            p = np.random.uniform(low=0, high=1, size=1) < self._epsilon
            self._epsilon -= self._epsilon * self._decay
        elif self.sampling_strategy == 'gibbs':
            lambda_, N = 1.4, self._tot_actions
            p = np.random.uniform(low=0, high=1, size=1) < boltzmann.rvs(lambda_, N, loc=0, size=1)
            self._tot_actions += 1
        elif self.sampling_strategy == 'thompson':
            p = None
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        if p:
            np.random.shuffle(sorted_bins)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}; {sorted_bins=}')
        fcs, ics, i = 0, 0, 0
        selected_bins = []
        while fcs < 1 or ics < 1:
            b = bins[sorted_bins[i][0], sorted_bins[i][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            i += 1
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {fcs=}; {ics=}; {i=}')
        return selected_bins

    def reset(self) -> None:
        self._epsilon = self._initial_epsilon
        self._buffer.clear()
        self._thompson_stats = {}
        self._estimator = None
        self._fitted = False
        self._tot_actions = 0

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            'tot_actions': self._tot_actions,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self._epsilon,
            'decay': self._decay,
            'buffer': self._buffer.to_json(),
            'diversity_weight': self.diversity_weight,
            'fitted': self._fitted,
            
            'thompson_stats': self._thompson_stats
        }
        j['estimator_name'] = self._estimator.__class__.__name__ if self._estimator else None
        if isinstance(self._estimator, KernelRidge):
            pass
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'LinearKernelEmitter':
        re = LinearKernelEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        re._tot_actions = my_args['tot_actions']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._epsilon = my_args['epsilon']
        re._decay = my_args['decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._n_features_context = my_args['n_features_context']
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
                re._estimator = KernelRidge()
                # TODO: load parameters
        
        re._thompson_stats = my_args['thompson_stats']
        
        return re


class RBFKernelEmitter(Emitter):
    def __init__(self,
                 epsilon: float = 0.2,
                 decay: float = 0.01) -> None:
        super().__init__()
        self.name = 'rbf-kernel-emitter'
        self.requires_pre = True
        
        self._initial_epsilon = epsilon
        self._epsilon = self._initial_epsilon
        self._decay = decay
        self._buffer = Buffer(merge_method=mean_merge)
        self._estimator: KNeighborsRegressor = None
        self._fitted = False
        
        self._tot_actions = 0
        self.sampling_strategy = 'thompson'  # or 'gibbs', 'thompson'
        self._thompson_stats = {}

    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self._estimator = KernelRidge(kernel='rbf').fit(X=xs, y=ys)
        logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self._estimator.score(xs, ys):.2%}')
        self._fitted = True
    
    def _get_valid_bins(self,
                        bins: 'np.ndarray[MAPBin]') -> npt.NDArray[np.float32]:
        valid = np.zeros_like(bins, dtype=np.uint8)
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    valid[i, j] = 1
        return valid
    
    def _get_bins_index(self,
                        bins: 'np.ndarray[MAPBin]',
                        normalize: bool = True) -> npt.NDArray[np.float32]:
        bin_idxs = np.zeros(shape=(bins.shape[0], bins.shape[1], 2))
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                bin_idxs[i, j, :] = [i, j]
        if normalize:
            bin_idxs[:, :, 0] = bin_idxs[:, :, 0] / bins.shape[0]
            bin_idxs[:, :, 1] = bin_idxs[:, :, 1] / bins.shape[1]
        return bin_idxs
    
    def _predict(self,
                 bins: 'np.ndarray[MAPBin]') -> Tuple[int, int]:
        mask = self._get_valid_bins(bins=bins)
        preferences = self._get_bins_index(bins=bins, normalize=True)
        mask3d = np.zeros(preferences.shape, dtype=bool)
        mask3d[:,:,:] = mask[:,:, np.newaxis] == 1
        predictions = self._estimator.predict(X=preferences[mask3d].reshape(-1, 2))
        logging.getLogger('mapelites').debug(f'[{__name__}._predict] {predictions=}')
        out = np.zeros_like(bins, dtype=np.float32)
        ii, jj = np.nonzero(mask)
        for i, j, v in zip(ii, jj, predictions):
            out[i, j] = v
        if self.sampling_strategy == 'thompson':
            for i, j in zip(ii, jj):
                n_i = i / bins.shape[0]
                n_j = j / bins.shape[1]
                scale_factor = np.random.beta(a=BETA_A + (self._thompson_stats[n_i, n_j][0] if (n_i, n_j) in self._thompson_stats else 1),
                                                b=BETA_B + (self._thompson_stats[n_i, n_j][1] if (n_i, n_j) in self._thompson_stats else 1 + self._tot_actions),
                                                size=1)
                out[i, j] *= scale_factor
        out = out.flatten()
        order = np.flip(np.argsort(out, axis=None))
        out = out[order]
        out_idxs = np.arange(len(out))[order][out != 0.]
        logging.getLogger('mapelites').debug(f'[{__name__}._predict] sorted predicted indices={out_idxs}; predicted values={out[out != 0.]}')
        return out_idxs
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        bounds: List[Tuple[float, float]] = kwargs['bounds']
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        bcs0 = np.cumsum([bounds[0][0]] + [b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([bounds[1][0]] + [b.bin_size[1] for b in bins[:, 0]])[:-1]
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                if bins[i, j].non_empty(pop='feasible'):
                    logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {(i, j)=}; x={np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]])}; dy={1. if (i, j) in idxs else 0.}')
                    self._buffer.insert(x=np.asarray([bcs0[i] / bounds[0][1], bcs1[j] / bounds[1][1]]),
                                        y=1. if (i, j) in idxs else 0.)
            n_i = i / bins.shape[0]
            n_j = j / bins.shape[1]
            if (i, j) in idxs:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [2, 1]
                else:
                    self._thompson_stats[n_i, n_j][0] += 1
            else:
                if (n_i, n_j) not in self._thompson_stats:
                    self._thompson_stats[n_i, n_j] = [1, 2]
                else:
                    self._thompson_stats[n_i, n_j][1] += 1
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        sorted_bins = np.transpose(np.unravel_index(self._predict(bins=bins), bins.shape))
        logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {sorted_bins=}')
        if self.sampling_strategy == 'epsilon_greedy':
            p = np.random.uniform(low=0, high=1, size=1) < self._epsilon
            self._epsilon -= self._epsilon * self._decay
        elif self.sampling_strategy == 'gibbs':
            lambda_, N = 1.4, self._tot_actions
            p = np.random.uniform(low=0, high=1, size=1) < boltzmann.rvs(lambda_, N, loc=0, size=1)
            self._tot_actions += 1
        elif self.sampling_strategy == 'thompson':
            p = None
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        if p:
            np.random.shuffle(sorted_bins)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}; {sorted_bins=}')
        fcs, ics, i = 0, 0, 0
        selected_bins = []
        while fcs < 1 or ics < 1:
            b = bins[sorted_bins[i][0], sorted_bins[i][1]]
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            i += 1
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {fcs=}; {ics=}; {i=}')
        return selected_bins

    def reset(self) -> None:
        self._epsilon = self._initial_epsilon
        self._buffer.clear()
        self._thompson_stats = {}
        self._estimator = None
        self._fitted = False
        self._tot_actions = 0

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            'tot_actions': self._tot_actions,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self._epsilon,
            'decay': self._decay,
            'buffer': self._buffer.to_json(),
            'diversity_weight': self.diversity_weight,
            'fitted': self._fitted,
            
            'thompson_stats': self._thompson_stats
        }
        j['estimator_name'] = self._estimator.__class__.__name__ if self._estimator else None
        if isinstance(self._estimator, KernelRidge):
            pass
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'RBFKernelEmitter':
        re = RBFKernelEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        re.diversity_weight = my_args['diversity_weight']
        re._tot_actions = my_args['tot_actions']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._epsilon = my_args['epsilon']
        re._decay = my_args['decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._n_features_context = my_args['n_features_context']
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
                re._estimator = KernelRidge()
                # TODO: load parameters
        
        re._thompson_stats = my_args['thompson_stats']
        
        return re


class HumanEmitter(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'human-emitter'
    
    def pick_bin(self, bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        return []

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


emitters = {
    'random-emitter': RandomEmitter,
    'optimising-emitter': OptimisingEmitter,
    'optimising-emitter-v2': OptimisingEmitterV2,
    'greedy-emitter': GreedyEmitter,
    'human-preference-matrix-emitter': HumanPrefMatrixEmitter,
    'contextual-bandit-emitter': ContextualBanditEmitter,
    'preference-bandit-emitter': PreferenceBanditEmitter,
    'human-emitter': HumanEmitter,
    'knn-emitter': KNNEmitter,
    'linear-kernel-emitter': LinearKernelEmitter,
    'rbf-kernel-emitter': RBFKernelEmitter
}
