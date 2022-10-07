import torch
import pyro.distributions as dist
import pyro
from pyro.infer import Predictive, NUTS, MCMC
from palmerpenguins import load_penguins

def linear_model(flipper_length, mass=None):

    sigma = pyro.sample('sigma', dist.HalfNormal(2000.))
    beta_0 = pyro.sample('beta_0', dist.Normal(0., 4000.))
    beta_1 = pyro.sample('beta_1', dist.Normal(0., 4000.))
    mu = pyro.deterministic('mu', beta_0 + beta_1 * flipper_length)

    with pyro.plate('plate'):   
        preds = pyro.sample('mass', dist.Normal(mu, sigma), obs=mass)  

def main():
    
    penguins = load_penguins()
    penguins.dropna(how='any', axis=0, inplace=True)
    adelie_mask = (penguins['species'] == 'Adelie')
    adelie_flipper_length = torch.from_numpy(penguins.loc[adelie_mask, 'flipper_length_mm'].values)
    adelie_mass = torch.from_numpy(penguins.loc[adelie_mask, 'body_mass_g'].values)

    kernel = NUTS(linear_model, adapt_step_size=True)
    mcmc_simple = MCMC(kernel, num_samples=500, warmup_steps=300, num_chains=4)
    mcmc_simple.run(flipper_length=adelie_flipper_length, mass=adelie_mass)

if __name__ == '__main__':
    main()