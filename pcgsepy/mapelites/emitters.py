from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import logging

import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPRegressor
from pcgsepy.config import BETA_A, BETA_B, CONTEXT_IDXS, CS_MAX_AGE, N_EPOCHS, USE_TORCH
from pcgsepy.mapelites.bin import MAPBin
from pcgsepy.mapelites.buffer import Buffer, mean_merge
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
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


# --- Preference-Learning Emitters implementations below ---
#     These are specific implementations of the general framework.


class HumanPrefMatrixEmitter(Emitter):
    def __init__(self,
                 delta: float = 1.,
                 decay: float = 5e-2,
                 sampling_strategy: str = 'gibbs',
                 epsilon: float = 9e-1,
                 tau: float = 1.,
                 sampling_decay: float = 1e-1) -> None:
        """Create a human preference-matrix emitter.

        Args:
            delta (float, optional): The preference increment. Defaults to `1`.
            decay (float, optional): The preference decay. Defaults to `5e-2`.
            sampling_strategy (str, optional): The sampling strategy. Valid values are `epsilon-greedy` and `gibbs`. Defaults to `gibbs`.
            epsilon (float, optional): The probability threshold used if sampling strategy is `epsilon-greedy`. Defaults to `9e-1`.
            tau (float, optional): The temperature used if sampling strategy is `gibbs`. Defaults to `1.`.
            sampling_decay (float, optional): The sampling decay. Defaults to `1e-1`.
        """
        super().__init__()
        self.name = 'human-preference-matrix-emitter'
        self.requires_init = True
        self.requires_post = True
        self.requires_pre = True
        
        self._delta = delta
        self._decay = decay
        self._prefs = None
        self._tot_actions = 0
        self._last_selected = []
        
        self.sampling_strategy = sampling_strategy  # epsilon_greedy or gibbs
        self.tau = tau
        self.epsilon = epsilon
        self._initial_tau = tau
        self._initial_epsilon = epsilon
        self.sampling_decay = sampling_decay
    
    def __repr__(self) -> str:
        return f'{self.name} {self.sampling_strategy} ({self.tau=};{self.epsilon=})'
    
    def _build_pref_matrix(self,
                           bins: 'np.ndarray[MAPBin]') -> None:
        """Build the preference matrix.

        Args:
            bins (np.ndarray[MAPBin]): The MAP-Elites bins.
        """
        self._prefs = np.zeros(shape=bins.shape, dtype=np.float16)
        for (i, j), b in np.ndenumerate(bins):
            self._prefs[i, j] = self._delta if b.non_empty(pop='feasible') or b.non_empty(pop='infeasbile') else 0.
        
    def _get_n_new_bins(self,
                        bins: 'np.ndarray[MAPBin]') -> int:
        n_new_bins = 0
        for (_, _), b in np.ndenumerate(bins):
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
    
    def _decay_preferences(self) -> None:
        self._prefs -= self._decay
        self._prefs[np.where(self._prefs < 0)] = 0
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._prefs is not None, 'Human-preference emitter has not been initialized! Preference matrix has not been set.'
        self._last_selected = []
        selected_bins = []
        non_empty = set([(i, j) for (i, j), b in np.ndenumerate(bins) if b.non_empty('feasible') and b.non_empty('infeasible')])
        valid_prefs = set([tuple(x) for x in np.argwhere(self._prefs > 0).tolist()])
        valid_idxs = list(non_empty.intersection(valid_prefs))
        if self.sampling_strategy == 'epsilon-greedy':
            p = (np.random.uniform(low=0, high=1, size=1) < self.epsilon)[0]
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}')
            self.epsilon -= self.sampling_decay * self.epsilon
            if p:
                np.random.shuffle(valid_idxs)
            else:
                valid_idxs.sort(key=lambda x: self._prefs[x], reverse=True)
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        elif self.sampling_strategy == 'gibbs':
            idxs = tuple(np.asarray(valid_idxs).transpose())
            logits = softmax(self._prefs[idxs] / self.tau)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {logits=}')
            self.tau -= self.sampling_decay * self.tau
            valid_bins = bins[idxs]
            sampled_bins = np.random.choice(valid_bins,
                                            size=len(valid_bins),
                                            replace=False,
                                            p=logits)
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        fcs, ics = 0, 0
        for b in sampled_bins:
            self._last_selected.append(np.argwhere(bins == b).tolist()[0])
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            if fcs > 0 and ics > 0:
                break
        return selected_bins
    
    def init_emitter(self,
                     **kwargs) -> None:
        assert self._prefs is None, f'{self.name} has already been initialized!'
        self._build_pref_matrix(bins=kwargs['bins'])
        
    def pre_step(self, **kwargs) -> None:
        assert self._prefs is not None, f'{self.name} has not been initialized! Preference matrix has not been set.'
        bins: np.ndarray[MAPBin] = kwargs['bins']
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
    
    def post_step(self,
                  bins: 'np.ndarray[MAPBin]') -> None:
        assert self._prefs is not None, f'{self.name} has not been initialized! Preference matrix has not been set.'
        self._decay_preferences()

    def reset(self) -> None:
        self._prefs = None
        self._tot_actions = 0
        self._tau = self._initial_tau
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'tot_actions': self._tot_actions,
            'decay': self._decay,
            'last_selected': self._last_selected,  # may need conversion tolist()
            'prefs': self._prefs.tolist(),
            'initial_tau': self._initial_tau,
            'tau': self.tau,
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self.epsilon,
            'sampling_strategy': self.sampling_strategy
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
        re._initial_epsilon = my_args['initial_epsilon']
        re.epsilon = my_args['epsilon']
        re._initial_tau = my_args['initial_tau']
        re.tau = my_args['tau']
        re.sampling_strategy = my_args['sampling_strategy']
        return re


class ContextualBanditEmitter(Emitter):
    def __init__(self,
                 n_features_context: int = len(CONTEXT_IDXS),
                 sampling_strategy: str = 'gibbs',
                 epsilon: float = 9e-1,
                 tau: float = 1.,
                 sampling_decay: float = 1e-1,
                 estimator: str = 'linear') -> None:
        """Create a contextual bandit emitter.

        Args:
            n_features_context (int, optional): The number of features in the solution context. Defaults to len(CONTEXT_IDXS).
            sampling_strategy (str, optional): The sampling strategy. Valid values are `epsilon-greedy` and `gibbs`. Defaults to 'gibbs'.
            epsilon (float, optional): The probability threshold value used if sampling strategy is 'epsilon-greedy'. Defaults to 9e-1.
            tau (float, optional): The temperature used if sampling strategy is `gibbs`. Defaults to `1.`.
            sampling_decay (float, optional): The sampling decay. Defaults to 1e-1.
            estimator (str, optional): The estimator type. Valid values are 'linear' and 'mlp'. Defaults to 'linear'
        """
        super().__init__()
        self.name: str = 'contextual-bandit-emitter'
        self.requires_pre: bool = True
        
        self._n_features_context: int = n_features_context
        self._buffer: Buffer = Buffer(merge_method=mean_merge)
        self._estimator: str = estimator
        self.estimator: Union[LinearRegression, NonLinearEstimator, MLPRegressor] = None
        self._fitted: bool = False
        
        self.sampling_strategy: str = sampling_strategy
        self.tau: float = tau
        self._initial_tau: float = tau
        self._initial_epsilon: float = epsilon
        self.epsilon: float = epsilon
        self.sampling_decay: float = sampling_decay
    
    def __repr__(self) -> str:
        return f'{self.name} {self._estimator} {self.sampling_strategy} ({self.tau=};{self.epsilon=})'
    
    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        if self._estimator == 'linear':
            self.estimator = LinearRegression().fit(X=xs, y=ys)
            logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        elif self._estimator == 'mlp':
            # if USE_TORCH:
            #     self.estimator = NonLinearEstimator(xshape=self._n_features_context,
            #                                         yshape=1)
            #     train_estimator(estimator=self._estimator,
            #                     xs=xs,
            #                     ys=ys,
            #                     n_epochs=20)
            # else:
                self.estimator = MLPRegressor(hidden_layer_sizes=(100, 100),
                                              activation='relu',
                                              alpha=1e-4,
                                              solver='lbfgs',
                                              verbose=1 if logging.getLogger('mapelites').level == logging.DEBUG else 0,
                                              max_iter=N_EPOCHS).fit(X=xs, y=ys)
                logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        else:
            raise ValueError(f'Unrecognized estimator type: {self._estimator}')
        self._fitted = True
    
    def _extract_bin_context(self,
                             b: MAPBin) -> npt.NDArray[np.float32]:
        return np.asarray(b.get_elite(population='feasible').representation)[CONTEXT_IDXS]
    
    def _predict(self,
                 bins: List[MAPBin]) -> Tuple[int, int]:
        context = [self._extract_bin_context(b=b) for b in bins]
        return self.estimator.predict(X=context)
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        for (i, j), b in np.ndenumerate(bins):
            if b.non_empty(pop='feasible'):
                self._buffer.insert(x=self._extract_bin_context(b),
                                    y=1. if (i, j) in idxs else 0.)
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        selected_bins = []
        valid_idxs = [(i, j) for (i, j), b in np.ndenumerate(bins) if b.non_empty('feasible') and b.non_empty('infeasible')]
        valid_bins = [bins[x] for x in valid_idxs]
        predicted_prefs = self._predict(bins=valid_bins)
        if self.sampling_strategy == 'epsilon-greedy':
            p = (np.random.uniform(low=0, high=1, size=1) < self.epsilon)[0]
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}')
            self.epsilon -= self.sampling_decay * self.epsilon
            if p:
                np.random.shuffle(valid_idxs)
            else:
                valid_idxs.sort(key=lambda x: predicted_prefs[np.argwhere(valid_idxs == x)], reverse=True)
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        elif self.sampling_strategy == 'gibbs':
            logits = softmax(predicted_prefs / self.tau)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {logits=}')
            self.tau -= self.sampling_decay * self.tau
            sampled_bins = np.random.choice(valid_bins,
                                            size=len(valid_bins),
                                            replace=False,
                                            p=logits)
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        fcs, ics = 0, 0
        for b in sampled_bins:
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            if fcs > 0 and ics > 0:
                break
        return selected_bins

    def reset(self) -> None:
        self.epsilon = self._initial_epsilon
        self.tau = self._initial_tau
        self._buffer.clear()
        self.estimator = None
        self._fitted = False
    
    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            'diversity_weight': self.diversity_weight,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self.epsilon,
            'initial_tau': self._initial_tau,
            'tau': self.tau,
            'sampling_decay': self.sampling_decay,
            'buffer': self._buffer.to_json(),
            'n_features_context': self._n_features_context,
            'fitted': self._fitted,
        }
        j['estimator_name'] = self._estimator
        if isinstance(self._estimator, LinearRegression):
            j['estimator_params'] = self.estimator.get_params(),
            j['estimator_coefs'] = self.estimator.coef_.tolist() if self._fitted else None,
            j['estimator_intercept'] = np.asarray(self.estimator.intercept_).tolist() if self._fitted else None,
        elif isinstance(self.estimator, MLPRegressor):
            j['coefs_'] = self.estimator.coefs_
            j['intercepts_']: self.estimator.intercepts_
            j['n_features_in_']: self.estimator.n_features_in_
            j['n_iter_']: self.estimator.n_iter_
            j['n_layers_']: self.estimator.n_layers_
            j['n_outputs_']: self.estimator.n_outputs_
            j['out_activation_']: self.estimator.out_activation_
        # elif USE_TORCH and isinstance(self._estimator, NonLinearEstimator):
        #     j['estimator_parameters'] = self._estimator.to_json()
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'ContextualBanditEmitter':
        re = ContextualBanditEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._initial_tau = my_args['initial_tau']
        re.epsilon = my_args['epsilon']
        re.tau = my_args['tau']
        re.sampling_decay = my_args['sampling_decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._n_features_context = my_args['n_features_context']
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
            # if my_args['estimator_name'] == 'NonLinearEstimator' and USE_TORCH and not USE_LINEAR_ESTIMATOR:
            #     re._estimator = NonLinearEstimator.from_json(my_args=my_args['estimator_parameters'])
            if my_args['estimator_name'] == 'linear':
                re._estimator = 'linear'
                re.estimator = LinearRegression()
                re.estimator.set_params(my_args['estimator_params'])
                if my_args['estimator_coefs'] is not None:
                    re.estimator.coef_ = np.asarray(my_args['estimator_coefs'])
                if my_args['estimator_intercept'] is not None:
                    re.estimator.intercept_ = np.asarray(my_args['estimator_intercept'])
            elif my_args['estimator_name'] == 'mlp':
                re._estimator = 'mlp'
                re.estimator = MLPRegressor()
                re.estimator.coefs_ = my_args['coefs_']
                re.estimator.intercepts_ = my_args['intercepts_']
                re.estimator.n_features_in_ = my_args['n_features_in_']
                re.estimator.n_iter_ = my_args['n_iter_']
                re.estimator.n_layers_ = my_args['n_layers_']
                re.estimator.n_outputs_ = my_args['n_outputs_']
                re.estimator.out_activation_ = my_args['out_activation_']
            else:
                raise ValueError(f'Unrecognized estimator name: {my_args["estimator_name"]}.')
        
        return re


class PreferenceBanditEmitter(Emitter):
    def __init__(self,
                 sampling_strategy: str = 'gibbs',
                 epsilon: float = 0.9,
                 tau: float = 1.,
                 sampling_decay: float = 0.01,
                 estimator: str = 'linear') -> None:
        """Create a preference bandit emitter.

        Args:
            sampling_strategy (str, optional): The sampling strategy. Valid values are `epsilon-greedy` and `gibbs`. Defaults to 'gibbs'.
            epsilon (float, optional): The probability threshold value used if sampling strategy is 'epsilon-greedy'. Defaults to 0.9.
            tau (float, optional): The temperature used if sampling strategy is `gibbs`. Defaults to 1..
            sampling_decay (float, optional): The sampling decay. Defaults to 0.01.
            estimator (str, optional):  The estimator type. Valid values are 'linear' and 'mlp'. Defaults to 'linear'.
        """
        super().__init__()
        self.name: str = 'preference-bandit-emitter'
        self.requires_pre: bool = True
        
        self._buffer: Buffer = Buffer(merge_method=mean_merge)
        self._estimator: str = estimator
        self.estimator: Union[LinearRegression, NonLinearEstimator, MLPRegressor] = None
        self._fitted: bool = False
        
        self.sampling_strategy: str = sampling_strategy
        self.tau: float = tau
        self._initial_tau: float = tau
        self.epsilon: float = epsilon
        self._initial_epsilon: float = epsilon
        self.sampling_decay: float = sampling_decay
    
    def __repr__(self) -> str:
        return f'{self.name} {self._estimator} {self.sampling_strategy} ({self.tau=};{self.epsilon=})'
    
    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        if self._estimator == 'linear':
            self.estimator = LinearRegression().fit(X=xs, y=ys)
            logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        elif self._estimator == 'mlp':
            # if USE_TORCH:
            #     self.estimator = NonLinearEstimator(xshape=self._n_features_context,
            #                                         yshape=1)
            #     train_estimator(estimator=self._estimator,
            #                     xs=xs,
            #                     ys=ys,
            #                     n_epochs=20)
            # else:
                self.estimator = MLPRegressor(hidden_layer_sizes=(100, 100),
                                              activation='relu',
                                              alpha=1e-4,
                                              solver='lbfgs',
                                              verbose=1 if logging.getLogger('mapelites').level == logging.DEBUG else 0,
                                              max_iter=N_EPOCHS).fit(X=xs, y=ys)
                logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        else:
            raise ValueError(f'Unrecognized estimator type: {self._estimator}')
        self._fitted = True
        
    def _predict(self,
                 bins: List[MAPBin]) -> Tuple[int, int]:
        xs = np.asarray([b.get_elite(population='feasible').b_descs for b in bins])
        return self.estimator.predict(xs)
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        bcs0 = np.cumsum([b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([b.bin_size[1] for b in bins[:, 0]])[:-1]
        for (i, j), b in np.ndenumerate(bins):
            if b.non_empty(pop='feasible'):
                self._buffer.insert(x=np.asarray([bcs0[i], bcs1[j]]),
                                    y=1. if (i, j) in idxs else 0.)
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        selected_bins = []
        valid_idxs = [(i, j) for (i, j), b in np.ndenumerate(bins) if b.non_empty('feasible') and b.non_empty('infeasible')]
        valid_bins = [bins[x] for x in valid_idxs]        
        predicted_prefs = self._predict(bins=valid_bins)
        if self.sampling_strategy == 'epsilon-greedy':
            p = (np.random.uniform(low=0, high=1, size=1) < self.epsilon)[0]
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}')
            self.epsilon -= self.sampling_decay * self.epsilon
            if p:
                np.random.shuffle(valid_idxs)
            else:
                valid_idxs.sort(key=lambda x: predicted_prefs[np.argwhere(valid_idxs == x)], reverse=True)
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        elif self.sampling_strategy == 'gibbs':
            logits = softmax(predicted_prefs / self.tau)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {logits=}')
            self.tau -= self.sampling_decay * self.tau
            sampled_bins = np.random.choice(valid_bins,
                                            size=len(valid_bins),
                                            replace=False,
                                            p=logits)
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        fcs, ics = 0, 0
        for b in sampled_bins:
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            if fcs > 0 and ics > 0:
                break
        return selected_bins

    def reset(self) -> None:
        self.epsilon = self._initial_epsilon
        self.tau = self._initial_tau
        self._buffer.clear()
        self._estimator = None
        self._fitted = False

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self.epsilon,
            'initial_tau': self._initial_tau,
            'tau': self.tau,
            'sampling_decay': self.sampling_decay,
            'buffer': self._buffer.to_json(),
            'fitted': self._fitted,
        }
        j['estimator_name'] = self._estimator
        if isinstance(self.estimator, LinearRegression):
            j['estimator_params'] = self.estimator.get_params(),
            j['estimator_coefs'] = self.estimator.coef_.tolist() if self._fitted else None,
            j['estimator_intercept'] = np.asarray(self.estimator.intercept_).tolist() if self._fitted else None,
        elif isinstance(self.estimator, MLPRegressor):
            j['coefs_'] = self.estimator.coefs_
            j['intercepts_']: self.estimator.intercepts_
            j['n_features_in_']: self.estimator.n_features_in_
            j['n_iter_']: self.estimator.n_iter_
            j['n_layers_']: self.estimator.n_layers_
            j['n_outputs_']: self.estimator.n_outputs_
            j['out_activation_']: self.estimator.out_activation_
        # elif USE_TORCH and isinstance(self._estimator, NonLinearEstimator):
        #     j['estimator_parameters'] = self._estimator.to_json()
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'PreferenceBanditEmitter':
        re = PreferenceBanditEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._initial_tau = my_args['initial_tau']
        re.epsilon = my_args['epsilon']
        re.tau = my_args['tau']
        re.sampling_decay = my_args['sampling_decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
            # if my_args['estimator_name'] == 'NonLinearEstimator' and USE_TORCH and not USE_LINEAR_ESTIMATOR:
            #     re._estimator = NonLinearEstimator.from_json(my_args=my_args['estimator_parameters'])
            if my_args['estimator_name'] == 'linear':
                re._estimator = 'linear'
                re.estimator = LinearRegression()
                re.estimator.set_params(my_args['estimator_params'])
                if my_args['estimator_coefs'] is not None:
                    re.estimator.coef_ = np.asarray(my_args['estimator_coefs'])
                if my_args['estimator_intercept'] is not None:
                    re.estimator.intercept_ = np.asarray(my_args['estimator_intercept'])
            elif my_args['estimator_name'] == 'mlp':
                re._estimator = 'mlp'
                re.estimator = MLPRegressor()
                re.estimator.coefs_ = my_args['coefs_']
                re.estimator.intercepts_ = my_args['intercepts_']
                re.estimator.n_features_in_ = my_args['n_features_in_']
                re.estimator.n_iter_ = my_args['n_iter_']
                re.estimator.n_layers_ = my_args['n_layers_']
                re.estimator.n_outputs_ = my_args['n_outputs_']
                re.estimator.out_activation_ = my_args['out_activation_']
            else:
                raise ValueError(f'Unrecognized estimator name: {my_args["estimator_name"]}.')
        
        return re


class KNEmitter(Emitter):
    def __init__(self,
                 sampling_strategy: str = 'gibbs',
                 epsilon: float = 9e-1,
                 tau: float = 1.,
                 sampling_decay: float = 1e-1) -> None:
        """Create a k-neighbours emitter.

        Args:
            sampling_strategy (str, optional): The sampling strategy. Valid values are `epsilon-greedy` and `gibbs`. Defaults to 'gibbs'.
            epsilon (float, optional): The probability threshold value used if sampling strategy is 'epsilon-greedy'. Defaults to 9e-1.
            tau (float, optional): The temperature used if sampling strategy is `gibbs`. Defaults to `1.`.
            sampling_decay (float, optional): The sampling decay. Defaults to 1e-1.
        """
        super().__init__()
        self.name: str = 'kn-emitter'
        self.requires_pre: bool = True
        
        self._buffer: Buffer = Buffer(merge_method=mean_merge)
        self.estimator: KNeighborsRegressor = None
        self._fitted: bool = False
        
        self.sampling_strategy: str = sampling_strategy
        self.tau: float = tau
        self._initial_tau: float = tau
        self._initial_epsilon: float = epsilon
        self.epsilon: float = epsilon
        self.sampling_decay: float = sampling_decay

    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self.estimator = KNeighborsRegressor(n_neighbors=5,
                                             leaf_size=30,
                                             weights='distance',
                                             p=2,
                                             metric='minkowski').fit(X=xs, y=ys)
        logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        self._fitted = True
    
    def __repr__(self) -> str:
        return f'{self.name} {self.sampling_strategy} ({self.tau=};{self.epsilon=})'
    
    def _predict(self,
                 bins: List[MAPBin]) -> Tuple[int, int]:
        xs = np.asarray([b.get_elite(population='feasible').b_descs for b in bins])
        return self.estimator.predict(xs)
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        bcs0 = np.cumsum([b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([b.bin_size[1] for b in bins[:, 0]])[:-1]
        for (i, j), b in np.ndenumerate(bins):
            if b.non_empty(pop='feasible'):
                self._buffer.insert(x=np.asarray([bcs0[i], bcs1[j]]),
                                    y=1. if (i, j) in idxs else 0.)
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        selected_bins = []
        valid_idxs = [(i, j) for (i, j), b in np.ndenumerate(bins) if b.non_empty('feasible') and b.non_empty('infeasible')]
        valid_bins: List[MAPBin] = [bins[x] for x in valid_idxs]        
        predicted_prefs = self._predict(bins=valid_bins)
        if self.sampling_strategy == 'epsilon-greedy':
            p = (np.random.uniform(low=0, high=1, size=1) < self.epsilon)[0]
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}')
            self.epsilon -= self.sampling_decay * self.epsilon
            if p:
                np.random.shuffle(valid_idxs)
            else:
                valid_idxs.sort(key=lambda x: predicted_prefs[np.argwhere(valid_idxs == x)], reverse=True)
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        elif self.sampling_strategy == 'gibbs':
            logits = softmax(predicted_prefs / self.tau)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {logits=}')
            self.tau -= self.sampling_decay * self.tau
            sampled_bins = np.random.choice(valid_bins,
                                            size=len(valid_bins),
                                            replace=False,
                                            p=logits)
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        fcs, ics = 0, 0
        for b in sampled_bins:
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            if fcs > 0 and ics > 0:
                break
        return selected_bins

    def reset(self) -> None:
        self.epsilon = self._initial_epsilon
        self.tau = self._initial_tau
        self._buffer.clear()
        self.estimator = None
        self._fitted = False

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self.epsilon,
            'initial_tau': self._initial_tau,
            'tau': self.tau,
            'sampling_decay': self.sampling_decay,
            'buffer': self._buffer.to_json(),
            'fitted': self._fitted,
        }
        j['estimator_name'] = 'kn'
        if isinstance(self.estimator, KNeighborsRegressor):
            # TODO: Save estimator parameters
            pass
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'KNEmitter':
        re = KNEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._initial_tau = my_args['initial_tau']
        re.epsilon = my_args['epsilon']
        re.tau = my_args['tau']
        re.sampling_decay = my_args['sampling_decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
                re._estimator = my_args['estimator_name']
                re.estimator = KNeighborsRegressor()
                # TODO: load parameters
        
        return re


class KernelEmitter(Emitter):
    def __init__(self,
                 sampling_strategy: str = 'gibbs',
                 epsilon: float = 0.9,
                 tau: float = 1.,
                 sampling_decay: float = 0.01,
                 estimator: str = 'linear') -> None:
        """Create a kernel-based emitter.

        Args:
            sampling_strategy (str, optional): The sampling strategy. Valid values are `epsilon-greedy` and `gibbs`. Defaults to 'gibbs'.
            epsilon (float, optional): The probability threshold value used if sampling strategy is 'epsilon-greedy'. Defaults to 0.9.
            tau (float, optional): The temperature used if sampling strategy is `gibbs`. Defaults to 1..
            sampling_decay (float, optional): The sampling decay. Defaults to 0.01.
            estimator (str, optional):  The estimator type. Valid values are 'linear' and 'rbf'. Defaults to 'linear'.
        """
        super().__init__()
        self.name: str = 'kernel-emitter'
        self.requires_pre: bool = True
        
        self._buffer: Buffer = Buffer(merge_method=mean_merge)
        self._estimator: str = estimator
        self.estimator: KernelRidge = None
        self._fitted: bool = False
        
        self.sampling_strategy: str = sampling_strategy
        self.tau: float = tau
        self._initial_tau: float = tau
        self.epsilon: float = epsilon
        self._initial_epsilon: float = epsilon
        self.sampling_decay: float = sampling_decay
    
    def __repr__(self) -> str:
            return f'{self.name} {self._estimator} {self.sampling_strategy} ({self.tau=};{self.epsilon=})'
    
    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self.estimator = KernelRidge(kernel=self._estimator,
                                     alpha=1.0).fit(X=xs, y=ys)
        logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        self._fitted = True
    
    def _predict(self,
                 bins: 'np.ndarray[MAPBin]') -> Tuple[int, int]:
        xs = np.asarray([b.get_elite(population='feasible').b_descs for b in bins])
        return self.estimator.predict(xs)
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        bcs0 = np.cumsum([b.bin_size[0] for b in bins[0, :]])[:-1]
        bcs1 = np.cumsum([b.bin_size[1] for b in bins[:, 0]])[:-1]
        for (i, j), b in np.ndenumerate(bins):
            if b.non_empty(pop='feasible'):
                self._buffer.insert(x=np.asarray([bcs0[i], bcs1[j]]),
                                    y=1. if (i, j) in idxs else 0.)
        self._fit()
        
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        selected_bins = []
        valid_idxs = [(i, j) for (i, j), b in np.ndenumerate(bins) if b.non_empty('feasible') and b.non_empty('infeasible')]
        valid_bins = [bins[x] for x in valid_idxs]        
        predicted_prefs = self._predict(bins=valid_bins)
        if self.sampling_strategy == 'epsilon-greedy':
            p = (np.random.uniform(low=0, high=1, size=1) < self.epsilon)[0]
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}')
            self.epsilon -= self.sampling_decay * self.epsilon
            if p:
                np.random.shuffle(valid_idxs)
            else:
                valid_idxs.sort(key=lambda x: predicted_prefs[np.argwhere(valid_idxs == x)], reverse=True)
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        elif self.sampling_strategy == 'gibbs':
            logits = softmax(predicted_prefs / self.tau)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {logits=}')
            self.tau -= self.sampling_decay * self.tau
            sampled_bins = np.random.choice(valid_bins,
                                            size=len(valid_bins),
                                            replace=False,
                                            p=logits)
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        fcs, ics = 0, 0
        for b in sampled_bins:
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            if fcs > 0 and ics > 0:
                break
        return selected_bins

    def reset(self) -> None:
        self.epsilon = self._initial_epsilon
        self.tau = self._initial_tau
        self._buffer.clear()
        self._estimator = None
        self._fitted = False

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self.epsilon,
            'initial_tau': self._initial_tau,
            'tau': self.tau,
            'sampling_decay': self.sampling_decay,
            'buffer': self._buffer.to_json(),
            'fitted': self._fitted,
        }
        j['estimator_name'] = self._estimator
        if isinstance(self.estimator, KernelRidge):
            # TODO: Save estimator parameters
            pass
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'KernelEmitter':
        re = KernelEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._initial_tau = my_args['initial_tau']
        re.epsilon = my_args['epsilon']
        re.tau = my_args['tau']
        re.sampling_decay = my_args['sampling_decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._fitted = my_args['fitted']
        
        if 'estimator_name' in my_args.keys():
                re._estimator = my_args['estimator_name']
                re.estimator = KernelRidge()
                # TODO: load parameters
        
        re._thompson_stats = my_args['thompson_stats']
        
        return re


class SimpleTabularEmitter(Emitter):
    def __init__(self,
                 sampling_strategy: str = 'thompson',
                 epsilon: float = 9e-1,
                 tau: float = 1.,
                 sampling_decay: float = 1e-1,
                 estimator: str = 'linear') -> None:
        """Create a simple tabular emitter.

        Args:
            sampling_strategy (str, optional): The sampling strategy. Valid values are `epsilon-greedy` and `gibbs`. Defaults to 'gibbs'.
            epsilon (float, optional): The probability threshold value used if sampling strategy is 'epsilon-greedy'. Defaults to 0.9.
            tau (float, optional): The temperature used if sampling strategy is `gibbs`. Defaults to 1..
            sampling_decay (float, optional): The sampling decay. Defaults to 0.01.
            estimator (str, optional):  The estimator type. Valid values are 'linear' and 'mlp'. Defaults to 'linear'.
        """
        super().__init__()
        self.name = 'simple-tabular-emitter'
        self.requires_pre: bool = True
        
        self._buffer: Buffer = Buffer(merge_method=mean_merge)
        self._estimator: str = estimator
        self.estimator: LinearRegression = None
        self._fitted: bool = False
        
        self.sampling_strategy: str = sampling_strategy
        self.tau: float = tau
        self._initial_tau: float = tau
        self.epsilon: float = epsilon
        self._initial_epsilon: float = epsilon
        self.sampling_decay: float = sampling_decay
        self.ts_priors = {}
        self._tot_actions = 0

    def __repr__(self) -> str:
        return f'{self.name} {self._estimator} {self.sampling_strategy} ({self.tau=};{self.epsilon=};{self.ts_priors=})'
    
    @ignore_warnings(category=ConvergenceWarning)
    def _fit(self) -> None:
        xs, ys = self._buffer.get()
        self.estimator = LinearRegression().fit(X=xs, y=ys)
        logging.getLogger('mapelites').debug(f'[{__name__}._fit] datapoints={len(xs)}; nonzero_count={len(np.nonzero(ys)[0])}; estimator_score={self.estimator.score(xs, ys):.2%}')
        self._fitted = True
        
    def _predict(self,
                 idxs: List[Tuple[int, int]]) -> Tuple[int, int]:
        return self.estimator.predict(idxs)
    
    def pre_step(self, **kwargs) -> None:
        bins: 'np.ndarray[MAPBin]' = kwargs['bins']
        idxs: List[Tuple[int, int]] = [*kwargs['selected_idxs'], *kwargs['expanded_idxs']]
        logging.getLogger('mapelites').debug(f'[{__name__}.pre_step] {idxs=}')
        for (i, j), b in np.ndenumerate(bins):
            if b.non_empty(pop='feasible'):
                self._buffer.insert(x=np.asarray([i, j]),
                                    y=1. if (i, j) in idxs else 0.)
                if (i, j) in self.ts_priors:
                    self.ts_priors[(i, j)]['a'] += 1 if (i, j) in idxs else 0
                    self.ts_priors[(i, j)]['b'] += 1
                else:
                    self.ts_priors[(i, j)] = {'a': BETA_A + 1 if (i, j) in idxs else BETA_A,
                                              'b': BETA_B}
        self._fit()
                
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
        assert self._fitted, f'{self.name} requires fitting and has not been fit yet!'
        selected_bins = []
        valid_idxs = [(i, j) for (i, j), b in np.ndenumerate(bins) if b.non_empty('feasible') and b.non_empty('infeasible')]
        valid_bins = [bins[x] for x in valid_idxs]        
        predicted_prefs = self._predict(idxs=valid_idxs)
        if self.sampling_strategy == 'epsilon-greedy':
            p = (np.random.uniform(low=0, high=1, size=1) < self.epsilon)[0]
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {p=}')
            self.epsilon -= self.sampling_decay * self.epsilon
            if p:
                np.random.shuffle(valid_idxs)
            else:
                valid_idxs.sort(key=lambda x: predicted_prefs[np.argwhere(valid_idxs == x)], reverse=True)
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        elif self.sampling_strategy == 'gibbs':
            logits = softmax(predicted_prefs / self.tau)
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {logits=}')
            self.tau -= self.sampling_decay * self.tau
            sampled_bins = np.random.choice(valid_bins,
                                            size=len(valid_bins),
                                            replace=False,
                                            p=logits)
        elif self.sampling_strategy == 'thompson':
            logging.getLogger('mapelites').debug(f'[{__name__}.pick_bin] {self.ts_priors=}')
            logits = [np.random.beta(a=self.ts_priors[idx]['a'] if idx in self.ts_priors else 1,
                                     b=self.ts_priors[idx]['b'] if idx in self.ts_priors else 1 + self._tot_actions,
                                     size=1) for idx in valid_idxs]
            valid_idxs = list(reversed([x for _, x in sorted(zip(logits, valid_idxs))]))
            sampled_bins = bins[tuple(np.asarray(valid_idxs).transpose())]
        else:
            raise Exception(f'Unknown sampling method for emitter: {self.sampling_strategy}.')
        fcs, ics = 0, 0
        for b in sampled_bins:
            fcs += len(b._feasible)
            ics += len(b._infeasible)
            selected_bins.append(b)
            if fcs > 0 and ics > 0:
                break
        self._tot_actions += 1
        return selected_bins
    
    def reset(self) -> None:
        self.epsilon = self._initial_epsilon
        self.tau = self._initial_tau
        self._buffer.clear()
        self._estimator = None
        self._fitted = False
        self.ts_priors = {}
        self._tot_actions = 0

    def to_json(self) -> Dict[str, Any]:
        j = {
            'name': self.name,
            'requires_init': self.requires_init,
            'requires_pre': self.requires_pre,
            'requires_post': self.requires_post,
            
            'initial_epsilon': self._initial_epsilon,
            'epsilon': self.epsilon,
            'initial_tau': self._initial_tau,
            'tau': self.tau,
            'sampling_decay': self.sampling_decay,
            'buffer': self._buffer.to_json(),
            'fitted': self._fitted,
            
            'tot_actions': self._tot_actions,
            'ts_priors': self.ts_priors
        }
        j['estimator_name'] = self._estimator
        if isinstance(self.estimator, LinearRegression):
            j['estimator_params'] = self.estimator.get_params(),
            j['estimator_coefs'] = self.estimator.coef_.tolist() if self._fitted else None,
            j['estimator_intercept'] = np.asarray(self.estimator.intercept_).tolist() if self._fitted else None,
        elif isinstance(self.estimator, MLPRegressor):
            j['coefs_'] = self.estimator.coefs_
            j['intercepts_']: self.estimator.intercepts_
            j['n_features_in_']: self.estimator.n_features_in_
            j['n_iter_']: self.estimator.n_iter_
            j['n_layers_']: self.estimator.n_layers_
            j['n_outputs_']: self.estimator.n_outputs_
            j['out_activation_']: self.estimator.out_activation_
        # elif USE_TORCH and isinstance(self._estimator, NonLinearEstimator):
        #     j['estimator_parameters'] = self._estimator.to_json()
        
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'SimpleTabularEmitter':
        re = SimpleTabularEmitter()
        re.name = my_args['name']
        re.requires_init = my_args['requires_init']
        re.requires_pre = my_args['requires_pre']
        re.requires_post = my_args['requires_post']
        
        re._initial_epsilon = my_args['initial_epsilon']
        re._initial_tau = my_args['initial_tau']
        re.epsilon = my_args['epsilon']
        re.tau = my_args['tau']
        re.sampling_decay = my_args['sampling_decay']
        re.buffer = Buffer.from_json(my_args['buffer'])
        re._fitted = my_args['fitted']
        
        re._tot_actions = my_args['_tot_actions']
        re.ts_priors = my_args['ts_priors']
        
        if 'estimator_name' in my_args.keys():
            # if my_args['estimator_name'] == 'NonLinearEstimator' and USE_TORCH and not USE_LINEAR_ESTIMATOR:
            #     re._estimator = NonLinearEstimator.from_json(my_args=my_args['estimator_parameters'])
            if my_args['estimator_name'] == 'linear':
                re._estimator = 'linear'
                re.estimator = LinearRegression()
                re.estimator.set_params(my_args['estimator_params'])
                if my_args['estimator_coefs'] is not None:
                    re.estimator.coef_ = np.asarray(my_args['estimator_coefs'])
                if my_args['estimator_intercept'] is not None:
                    re.estimator.intercept_ = np.asarray(my_args['estimator_intercept'])
            elif my_args['estimator_name'] == 'mlp':
                re._estimator = 'mlp'
                re.estimator = MLPRegressor()
                re.estimator.coefs_ = my_args['coefs_']
                re.estimator.intercepts_ = my_args['intercepts_']
                re.estimator.n_features_in_ = my_args['n_features_in_']
                re.estimator.n_iter_ = my_args['n_iter_']
                re.estimator.n_layers_ = my_args['n_layers_']
                re.estimator.n_outputs_ = my_args['n_outputs_']
                re.estimator.out_activation_ = my_args['out_activation_']
            else:
                raise ValueError(f'Unrecognized estimator name: {my_args["estimator_name"]}.')
        
        return re
    

class HumanEmitter(Emitter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'human-emitter'
    
    def pick_bin(self,
                 bins: 'np.ndarray[MAPBin]') -> List[MAPBin]:
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
    'kn-emitter': KNEmitter,
    'kernel-emitter': KernelEmitter,
}
