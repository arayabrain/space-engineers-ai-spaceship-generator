from typing import Any, Dict, List, Union
import torch as th
import numpy as np
from pcgsepy.config import EPSILON_F

from pcgsepy.fi2pop.utils import MLPEstimator


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
        qe.load_state_dict(eval(my_args['model_params']))
        qe.load_state_dict(eval(my_args['optimizer']))
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
        out = th.F.elu(self.l1(x))
        out = th.F.elu(self.l2(out))
        out = th.F.elu(self.l3(out))
        return th.clamp(out, EPSILON_F, 1)

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


def train_estimator(estimator: Union[MLPEstimator, QuantileEstimator],
                    xs: List[List[float]],
                    ys: List[List[float]],
                    n_epochs: int = 50):
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
