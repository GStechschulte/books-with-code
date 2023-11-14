from typing import NamedTuple, Union

import gpytorch
import torch


class GPData(NamedTuple):
    train_x: torch.Tensor
    train_y: torch.Tensor
    xs: Union[torch.Tensor, None] = None
    ys: Union[torch.Tensor, None] = None


def accuracy_fn(x):
    """
    Simulates the accuracy surface of a support-vector machine (SVM) in 
    a hyperparameter tuning task.
     
    x[:, 0] = penalty parameter $c$
    x[:, 1] = RBF kernel parameter $\gamma$. 
    """
    return (
        torch.sin(5 * x[..., 0] / 2 - 2.5) 
        * torch.cos(2.5 - 5 * x[..., 1]) 
        + (5 * x[..., 1] / 2 + 0.5) ** 2 / 10) / 5 + 0.2


def forrester_fn(x):
    y = -((x + 1) ** 2) * torch.sin(2 * x + 2) / 5 + 1
    return y.squeeze(-1)


def flight_objective_fn(x):
    X_copy = x.detach().clone()
    X_copy[:, [2, 3]] = 1 - X_copy[:, [2, 3]]
    X_copy = X_copy * 10 - 5

    return -0.005 * (X_copy ** 4 - 16 * X_copy ** 2 + 5 * X_copy).sum(dim=-1) + 3