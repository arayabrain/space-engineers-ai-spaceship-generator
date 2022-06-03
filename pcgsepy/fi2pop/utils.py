from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler

from ..config import EPSILON_F, POP_SIZE, RESCALE_INFEAS_FITNESS
from ..evo.genops import EvoException, crossover, mutate, roulette_wheel_selection
from ..lsystem.constraints import ConstraintLevel, ConstraintTime
from ..lsystem.lsystem import LSystem
from ..lsystem.parser import HLtoMLTranslator, LLParser
from ..lsystem.solution import CandidateSolution
from ..structure import Structure


def subdivide_solutions(lcs: List[CandidateSolution],
                        lsystem: LSystem,
                        ) -> None:
    """Assign feasibility flag to the solutions.

    Args:
        lcs (List[CandidateSolution]): The list of solutions.
        lsystem (LSystem): The L-system used to check feasibility for.
    """
    lsystem.hl_solver.set_constraints(cs=lsystem.all_hl_constraints)
    lsystem.ll_solver.set_constraints(cs=lsystem.all_ll_constraints)
    for cs in lcs:
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            sat = lsystem.hl_solver._check_constraints(cs=cs,
                                                       when=t,
                                                       keep_track=True)
            cs.is_feasible &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
            cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
            cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]            
        if cs.is_feasible:
            if cs.ll_string == '':
                ml_string = lsystem.hl_solver.translator.transform(string=cs.string)
                cs.ll_string = LLParser(rules=lsystem.hl_solver.ll_rules).expand(string=ml_string)
            for t in [ConstraintTime.DURING, ConstraintTime.END]:
                sat = lsystem.ll_solver._check_constraints(cs=cs,
                                                           when=t,
                                                           keep_track=True)
                cs.is_feasible &= sat[ConstraintLevel.HARD_CONSTRAINT][0]
                cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
                cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]


def create_new_pool(population: List[CandidateSolution],
                    generation: int,
                    n_individuals: int = POP_SIZE,
                    minimize: bool = False) -> List[CandidateSolution]:
    """Create a new pool of solutions.

    Args:
        population (List[CandidateSolution]): Initial population of solutions.
        generation (int): Current generation number.
        n_individuals (int, optional): The number of individuals in the pool. Defaults to POP_SIZE.
        minimize (bool, optional): Whether to minimize or maximize the fitness. Defaults to False.

    Raises:
        EvoException: If same parent is picked twice for crossover.

    Returns:
        List[CandidateSolution]: The pool of new solutions.
    """
    pool = []
    while len(pool) < n_individuals:
        # fitness-proportionate selection
        p1 = roulette_wheel_selection(pop=population,
                                      minimize=minimize)
        
        new_pop = []
        new_pop[:] = population[:]
        new_pop.remove(p1)
        
        p2 = roulette_wheel_selection(pop=new_pop,
                                      minimize=minimize)
        if p1 != p2:
            # crossover
            o1, o2 = crossover(a1=p1, a2=p2, n_childs=2)
            # set parents
            o1.parents = [p1, p2]
            o2.parents = [p1, p2]
            for o in [o1, o2]:
                # mutation
                mutate(cs=o, n_iteration=generation)
                if o not in pool:
                    pool.append(o)
        else:
            raise EvoException('Picked same parents, this should never happen.')
    return pool


def reduce_population(population: List[CandidateSolution],
                      to: int,
                      minimize: bool = False) -> List[CandidateSolution]:
    """Order and reduce a population to a given size.

    Args:
        population (List[CandidateSolution]): The population.
        to (int): The desired population size.
        minimize (bool, optional): Whether to order for descending (False) or ascending (True) values. Defaults to False.

    Returns:
        List[CandidateSolution]: The ordered and culled population.
    """
    population.sort(key=lambda x: x.c_fitness,
                    reverse=not minimize)
    return population[:to]


class MLPEstimator(nn.Module):
    def __init__(self,
                 xshape: int,
                 yshape: int):
        """Create the MLPEstimator.

        Args:
            xshape (int): The number of dimensions in input.
            yshape (int): The number of dimensions in output.
        """
        super(MLPEstimator, self).__init__()
        self.xshape = xshape
        self.yshape = yshape
        self.l1 = nn.Linear(xshape, xshape*2)
        self.l2 = nn.Linear(xshape*2, int(xshape*2 / 3))
        self.l3 = nn.Linear(int(xshape*2 / 3), yshape)

        self.optimizer = th.optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()
        self.is_trained = False

    def forward(self, x):
        out = F.elu(self.l1(x))
        out = F.elu(self.l2(out))
        out = F.elu(self.l3(out))
        return th.clamp(out, 1e-5, 1)

    def save(self,
             fname: str):
        """Save the current model to file.

        Args:
            fname (str): The filename.
        """
        with open(f'{fname}.pth', 'wb') as f:
            th.save({
                'model_params': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'is_trained': self.is_trained
            }, f)

    def load(self,
             fname: str):
        """Load the parameters for the model from file.

        Args:
            fname (str): The filename.
        """
        with open(f'{fname}.pth', 'rb') as f:
            prev = th.load(f)
            self.load_state_dict(prev['model_params'])
            self.optimizer.load_state_dict(prev['optimizer'])
            self.is_trained = prev['is_trained']
        
    def to_json(self) -> Dict[str, Any]:
        return {
            'xshape': self.xshape,
            'yshape': self.yshape,
            'is_trained': self.is_trained,
            'model_params': str(self.state_dict()),
            'optimizer': str(self.optimizer.state_dict()),
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'MLPEstimator':
        mlpe = MLPEstimator(xhsape=my_args['xshape'],
                            yshape=my_args['yshape'])
        mlpe.is_trained = my_args['is_trained']
        mlpe.load_state_dict(eval(my_args['model_params']))
        mlpe.load_state_dict(eval(my_args['optimizer']))
        return mlpe

def train_estimator(estimator: MLPEstimator,
                    xs: List[List[float]],
                    ys: List[List[float]],
                    n_epochs: int = 20):
    """Train the MLP estimator.

    Args:
        estimator (MLPEstimator): The estimator to train.
        xs (List[List[float]]): The low-dimensional input vector.
        ys (List[List[float]]): The output vector (mean offsprings fitness).
        n_epochs (int, optional): The number of epochs to train for. Defaults to 5.
    """
    xs = th.tensor(xs).float().squeeze(1)
    ys = th.tensor(ys).float().unsqueeze(1)
    losses = []
    for _ in range(n_epochs):
        estimator.optimizer.zero_grad()
        out = estimator(xs)
        loss = estimator.criterion(out, ys)
        losses.append(loss.item())
        loss.backward()
        estimator.optimizer.step()
    
    with open('estimators_perf.log', 'a') as f:
        f.write(f'{type(estimator).__name__} train losses: {losses}\n')
    
    estimator.is_trained = True


class GaussianEstimator:
    def __init__(self,
                 bound: str,
                 kernel: Any,
                 max_f: float,
                 min_f: float = 0,
                 alpha: float = 1e-10,
                 normalize_y: bool = False) -> None:
        self.bound = bound
        self.max_f = max_f
        self.min_f = min_f
        self.gpr = GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            normalize_y=normalize_y)
        self.is_trained = False
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> None:
        self.gpr.fit(X, y)
        self.is_trained = True
    
    def predict(self,
                x: np.ndarray) -> float:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        y_mean, y_std = self.gpr.predict(x, return_std=True)
        if self.bound == 'upper':
            f = (y_mean[0] + y_std[0]) / self.max_f
        elif self.bound == 'lower':
            f = max(y_mean[0] - y_std[0], self.min_f)
        else:
            raise NotImplementedError(f'Unrecognized bound ({self.bound}) encountered in GaussianEstimator.')
        return f


class DimensionalityReducer:
    def __init__(self,
                 n_components: int,
                 max_dims: int):
        """Generate the DimensionalityReducer

        Args:
            n_components (int): The number of dimensions to reduce to.
            max_dims (int): The maximum number of dimensions to consider in input.
        """
        self.pca = PCA(n_components=n_components,
                       svd_solver='full')  # TODO: Test with KernelPCA and SparsePCA as well
        self.scaler = StandardScaler()
        self.max_dims = max_dims

    def fit(self,
            xs: List[Structure]):
        """Fit the DimensionalityReducer to the data.

        Args:
            xs (List[Structure]): A list of Structures.
        """
        arrs = []
        for x in xs:
            arr = x.as_grid_array()
            arr = arr / np.linalg.norm(arr)
            arrs.append(arr.flatten())
        s = max(self.max_dims, max([x.shape[0] for x in arrs]))
        self.max_dims = s
        to_fit = np.zeros(shape=(len(arrs), s))
        for i, x in enumerate(arrs):
            to_fit[i, :min(s, x.shape[0])] = x[:min(s, x.shape[0])]
        self.scaler.fit(to_fit)
        to_fit = self.scaler.transform(to_fit)
        self.pca = self.pca.fit(to_fit)

    def reduce_dims(self,
                    s: Structure) -> List[float]:
        """Reduce the dimensions of the Structure.

        Args:
            s (Structure): The Structure.

        Returns:
            List[float]: The low-dimensional vector of the Structure (`len(_) = n_components`).
        """
        arr = s.as_grid_array()
        arr = arr / np.linalg.norm(arr)
        arr = arr.flatten()
        x = np.zeros(shape=(1, self.max_dims))
        x[0, :min(self.max_dims, arr.shape[0])] = arr[:min(self.max_dims, arr.shape[0])]
        x = self.scaler.transform(x)
        return self.pca.transform(x).tolist()


def prepare_dataset(f_pop: List[CandidateSolution]) -> Tuple[List[List[float]]]:
    """Prepare the dataset for the estimator.

    Args:
        f_pop (List[CandidateSolution]): The Feasible population.

    Returns:
        Tuple[List[List[float]]]: Inputs and labels to use during training.
    """
    xs, ys = [], []
    for cs in f_pop:
        y = cs.c_fitness
        for parent in cs.parents:
            if not parent.is_feasible:
                x = parent.representation
                parent.n_feas_offspring += 1
                xs.append(x)
                ys.append(y if not RESCALE_INFEAS_FITNESS else y * (EPSILON_F + (parent.n_feas_offspring / parent.n_offspring)))
    return xs, ys
