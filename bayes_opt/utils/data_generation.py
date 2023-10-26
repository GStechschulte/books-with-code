import gpytorch
import torch

def accuracy_func(x):
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