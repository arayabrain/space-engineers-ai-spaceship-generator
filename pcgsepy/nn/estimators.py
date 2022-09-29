from typing import Any, Dict, List, Tuple, Union

import ast
import warnings
import numpy as np
from pcgsepy.config import EPSILON_F, RESCALE_INFEAS_FITNESS, USE_TORCH
from pcgsepy.lsystem.solution import CandidateSolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


if USE_TORCH:
    import torch as th
    import torch.nn.functional as F

    def quantile_loss(predicted: th.Tensor,
                    target: th.Tensor) -> th.Tensor:
        """Compute quantile loss on 0.05, 0.5, and 0.95 quantiles.

        Inspired by Keras quantile loss:
        ```
        e = y_p - y
        return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))
        ```

        Args:
            predicted (th.Tensor): The predicted tensor.
            target (th.Tensor): The target tensor.

        Returns:
            th.Tensor: The quantile loss, expressed as mean of the three quantiles losses.
        """
        assert not target.requires_grad
        q1, q2, q3 = 0.05, 0.5, 0.95
        e1 = predicted[:, 0] - target
        e2 = predicted[:, 1] - target
        e3 = predicted[:, 2] - target
        eq1 = th.max(q1 * e1, (q1 - 1) * e1)
        eq2 = th.max(q2 * e2, (q2 - 1) * e2)
        eq3 = th.max(q3 * e3, (q3 - 1) * e3)

        return th.mean(eq1 + eq2 + eq3)

    class QuantileEstimator(th.nn.Module):
        def __init__(self,
                    xshape: int,
                    yshape: int):
            """Create the QuantileEstimator.

            Args:
                xshape (int): The number of dimensions in input.
                yshape (int): The number of dimensions in output.
            """
            super(QuantileEstimator, self).__init__()
            self.xshape = xshape
            self.yshape = yshape
            self.q1_l = th.nn.Linear(32, yshape)
            self.q2_l = th.nn.Linear(32, yshape)
            self.q3_l = th.nn.Linear(32, yshape)

            self.net = th.nn.Sequential(
                th.nn.Linear(xshape, 16),
                th.nn.LeakyReLU(),
                th.nn.Linear(16, 32),
                th.nn.LeakyReLU(),
            )

            self.optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
            self.criterion = quantile_loss
            self.is_trained = False
            self.train_losses = []

        def forward(self, x):
            out = self.net(x)
            yq1 = self.q1_l(out)
            yq2 = self.q2_l(out)
            yq3 = self.q3_l(out)
            out = th.cat((yq1, yq2, yq3), dim=1)
            return th.sigmoid(out)

        def predict(self,
                    x: np.ndarray) -> float:
            with th.no_grad():
                return self.forward(th.tensor(x).float().unsqueeze(0)).numpy()[0]
        
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
        def from_json(my_args: Dict[str, Any]) -> 'QuantileEstimator':
            qe = QuantileEstimator(xhsape=my_args['xshape'],
                                    yshape=my_args['yshape'])
            qe.is_trained = my_args['is_trained']
            qe.load_state_dict(ast.literal_eval(my_args['model_params']))
            qe.load_state_dict(ast.literal_eval(my_args['optimizer']))
            return qe
    
    class MLPEstimator(th.nn.Module):
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
            self.l1 = th.nn.Linear(xshape, xshape * 2)
            self.l2 = th.nn.Linear(xshape * 2, int(xshape * 2 / 3))
            self.l3 = th.nn.Linear(int(xshape * 2 / 3), yshape)

            self.optimizer = th.optim.Adam(self.parameters())
            self.criterion = th.nn.MSELoss()
            self.is_trained = False
            self.train_losses = []

        def forward(self, x):
            out = F.elu(self.l1(x))
            out = F.elu(self.l2(out))
            out = F.elu(self.l3(out))
            return th.clamp(out, EPSILON_F, 1)

        def predict(self,
                    x: np.ndarray) -> float:
            with th.no_grad():
                return self.forward(th.tensor(x).float()).numpy()[0]
        
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
            mlpe.load_state_dict(ast.literal_eval(my_args['model_params']))
            mlpe.load_state_dict(ast.literal_eval(my_args['optimizer']))
            return mlpe

    class NonLinearEstimator(th.nn.Module):
        def __init__(self,
                    xshape: int,
                    yshape: int):
            """Create the NonLinearEstimator.

            Args:
                xshape (int): The number of dimensions in input.
                yshape (int): The number of dimensions in output.
            """
            super(NonLinearEstimator, self).__init__()
            self.xshape = xshape
            self.yshape = yshape
            self.l1 = th.nn.Linear(xshape, xshape * 2)
            self.l2 = th.nn.Linear(xshape * 2, int(xshape * 2 / 3))
            self.l3 = th.nn.Linear(int(xshape * 2 / 3), yshape)

            self.optimizer = th.optim.Adam(self.parameters())
            self.criterion = th.nn.CrossEntropyLoss()
            self.is_trained = False
            self.train_losses = []

        def forward(self, x):
            out = F.elu(self.l1(x))
            out = F.elu(self.l2(out))
            out = F.elu(self.l3(out))
            return th.clamp(out, 0, 1)
        
        def predict(self,
                    X: np.ndarray) -> float:
            with th.no_grad():
                return self.forward(th.tensor(X).float()).squeeze().numpy()
        
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
        def from_json(my_args: Dict[str, Any]) -> 'NonLinearEstimator':
            nle = NonLinearEstimator(xhsape=my_args['xshape'],
                                    yshape=my_args['yshape'])
            nle.is_trained = my_args['is_trained']
            nle.load_state_dict(ast.literal_eval(my_args['model_params']))
            nle.load_state_dict(ast.literal_eval(my_args['optimizer']))
            return nle

    def train_estimator(estimator: Union[MLPEstimator, QuantileEstimator],
                        xs: List[List[float]],
                        ys: List[List[float]],
                        n_epochs: int = 20):
        """Train the MLP estimator.

        Args:
            estimator (Union[MLPEstimator, QuantileEstimator]): The estimator to train.
            xs (List[List[float]]): The low-dimensional input vector.
            ys (List[List[float]]): The output vector (mean offsprings fitness).
            n_epochs (int, optional): The number of epochs to train for. Defaults to 20.
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
        
        estimator.train_losses.append(losses[-1])
        
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
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.gpr = GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            normalize_y=normalize_y)
        self.is_trained = False
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit(self,
            xs: np.ndarray,
            ys: np.ndarray) -> None:
        self.gpr.fit(xs, ys)
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
         
    def to_json(self) -> Dict[str, Any]:
        return {
            'bound': self.bound,
            'max_f': self.max_f,
            'min_f': self.min_f,
            'is_trained': self.is_trained,
            'gpr': self.gpr.get_params()
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'GaussianEstimator':
        gpe = GaussianEstimator(bound=my_args['bound'],
                                max_f=my_args['max_f'],
                                min_f=my_args['min_f'],
                                kernel=None)
        gpe.is_trained = my_args['is_trained']
        gpe.gpr.set_params(my_args['gpr'])
        return gpe


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
