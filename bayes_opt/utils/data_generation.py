from typing import NamedTuple

import gpytorch
import torch

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


class GPData(NamedTuple):
    train_x: torch.Tensor
    train_y: torch.Tensor
    xs: torch.Tensor
    ys: torch.Tensor