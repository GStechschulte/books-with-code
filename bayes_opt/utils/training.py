import gpytorch
import torch
from tqdm import tqdm


def fit_gp_model(GP, data, num_train_iters=500):
    noise = 1e-4
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(data.train_x, data.train_y, likelihood) 
    model.likelihood.noise = noise
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    model.train()
    likelihood.train()

    for _ in tqdm(range(num_train_iters)):
        optimizer.zero_grad()
        
        output = model(data.train_x)
        loss = -mll(output, data.train_y)
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    likelihood.eval()

    return model, likelihood