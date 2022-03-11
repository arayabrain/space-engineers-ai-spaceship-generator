from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, KernelPCA, SparsePCA

from pcgsepy.lsystem.solution import CandidateSolution
from pcgsepy.structure import Structure

from ..config import POP_SIZE
from ..evo.genops import crossover, mutate, roulette_wheel_selection
from ..lsystem.constraints import ConstraintLevel, ConstraintTime
from ..lsystem.lsystem import LSystem
from ..lsystem.parser import HLtoMLTranslator, LLParser


def subdivide_solutions(lcs: List[CandidateSolution],
                        lsystem: LSystem) -> None:
    lsystem.hl_solver.set_constraints(cs=lsystem.all_hl_constraints)
    lsystem.ll_solver.set_constraints(cs=lsystem.all_ll_constraints)
    for cs in lcs:
        for t in [ConstraintTime.DURING, ConstraintTime.END]:
            sat = lsystem.hl_solver._check_constraints(cs=cs,
                                                       when=t,
                                                       keep_track=True)
            cs.is_feasible = sat[ConstraintLevel.HARD_CONSTRAINT][0]
            cs.ncv += sat[ConstraintLevel.HARD_CONSTRAINT][1]
            cs.ncv += sat[ConstraintLevel.SOFT_CONSTRAINT][1]
        if cs.is_feasible:
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
    pool = []

    while len(pool) < n_individuals:
        # fitness-proportionate selection
        p1 = roulette_wheel_selection(pop=population,
                                      minimize=minimize)
        p2 = roulette_wheel_selection(pop=population,
                                      minimize=minimize)
        # crossover
        o1, o2 = crossover(a1=p1, a2=p2, n_childs=2)
        
        o1.parents = [p1, p2]
        o2.parents = [p1, p2]

        for o in [o1, o2]:
            # mutation
            mutate(cs=o, n_iteration=generation)
            if o not in pool:
                pool.append(o)

    return pool


def reduce_population(population: List[CandidateSolution],
                      to: int,
                      minimize: bool = False) -> List[CandidateSolution]:
    population.sort(key=lambda x: x.c_fitness if x.is_feasible else x.ncv,
                    reverse=minimize)
    return population[-to:]


class MLPEstimator(nn.Module):
    def __init__(self,
                 xshape,
                 yshape):
        super(MLPEstimator, self).__init__()
        self.l1 = nn.Linear(xshape, xshape*2)
        self.l2 = nn.Linear(xshape*2, int(xshape*2 / 3))
        self.l3 = nn.Linear(int(xshape*2 / 3), yshape)
        self.is_trained = False
        
        self.optimizer = th.optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        out = F.elu(self.l1(x))
        out = F.elu(self.l2(out))
        out = F.elu(self.l3(out))
        return th.clamp(out, 1e-5, 1)
    
    # TODO: Implement save/load methods
    def save(self, fname):
        pass
    
    @staticmethod
    def load(fname) -> 'MLPEstimator':
        return None


def train_estimator(estimator: MLPEstimator,
                    xs,
                    ys,
                    n_epochs: int = 5):
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
    estimator.is_trained = True


class DimensionalityReducer:
    def __init__(self,
                 n_components,
                 max_dims):
        self.pca = PCA(n_components=n_components,
                       svd_solver='full')  # TODO: Test with KernelPCA and SparsePCA as well
        self.scaler = StandardScaler()
        self.max_dims = max_dims
    
    def _fit_scaler(self,
                    xs: np.ndarray):
        self.scaler = self.scaler.fit(xs)
    
    def fit(self,
            xs: List[Structure]):
        arrs = []
        for x in xs:
            arr = x.as_grid_array()
            arr = arr / np.linalg.norm(arr)
            arrs.append(arr.flatten())
        
        s = max(self.max_dims, max([x.shape[0] for x in arrs]))
        to_fit = np.zeros(shape=(len(arrs), s))
        for i, x in enumerate(arrs):
            to_fit[i,:min(s, x.shape[0])] = x[:min(s, x.shape[0])]
        self._fit_scaler(xs=to_fit)
        self.pca = self.pca.fit(to_fit)
    
    def reduce_dims(self,
                    s: Structure) -> List[float]:
        arr = s.as_grid_array()
        arr = arr / np.linalg.norm(arr)
        arr = arr.flatten()
        s = max(self.max_dims, arr.shape[0])
        x = np.zeros(shape=(1, s))
        x[0, :min(s, arr.shape[0])] = arr[:min(s, arr.shape[0])]
        return self.pca.transform(x)


def prepare_dataset(f_pop: List[CandidateSolution],
                    reducer) -> Tuple[List[List[float]]]:
    xs, ys = [], []
    for cs in f_pop:
        y = cs.c_fitness
        for parent in cs.parents:
            if not parent.is_feasible:
                x = reducer.reduce_dims(parent._content).tolist()
                if x in xs:
                    curr_y = ys[xs.index(x)]
                    ys[xs.index(x)] = (y + curr_y) / 2
                else:
                    xs.append(x)
                    ys.append(y)
    return xs, ys