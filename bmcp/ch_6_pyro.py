from utilities import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import arviz as az
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, NUTS, MCMC
from pyro.optim import Adam
from pyro.infer.mcmc.util import summary
import os

def gam(fourier_x, mean_co2, A, s, t, obs=None):

    N, P = fourier_x.shape
    num_changepoints = 12

    k = pyro.sample('k', dist.HalfNormal(10.))
    m = pyro.sample('m', dist.Normal(float(mean_co2), scale=5.))
    tau = pyro.sample('tau', dist.HalfNormal(10.))

    with pyro.plate('season', P):
        beta = pyro.sample('beta', dist.Normal(0., 1.))
        delta = pyro.sample('delta', dist.Laplace(0., tau))

    seasonality = torch.matmul(fourier_x.type(dtype=torch.float64), beta.type(dtype=torch.float64))
    growth_rate = k + torch.matmul(A.type(dtype=torch.float64), delta.type(dtype=torch.float64))
    gamma = -s * delta.type(dtype=torch.float64).type(dtype=torch.float64)
    offset = m + torch.matmul(A, gamma)
    trend = torch.matmul(growth_rate, t.type(dtype=torch.float64)) + offset

    y_hat = torch.add(seasonality, trend)
    noise_sigma = pyro.sample('noise_sigma', dist.HalfNormal(5.))
    
    with pyro.plate('obs', N):
        observed = pyro.sample('output', dist.Normal(y_hat, noise_sigma), obs=obs)


def gen_fourier_basis(t, p=365.25, n=3):

    x = 2 * torch.pi * (torch.arange(n) + 1) * t[:, None] / p

    return torch.concatenate((torch.cos(x), torch.sin(x)), axis=1)


def main():

    co2_by_month = pd.read_csv(os.path.abspath('.') + '/bmcp/data/monthly_mauna_loa_co2.csv')
    co2_by_month["date_month"] = pd.to_datetime(co2_by_month["date_month"])
    co2_by_month["CO2"] = co2_by_month["CO2"].astype(np.float32)
    co2_by_month.set_index("date_month", drop=True, inplace=True)

    num_forecast_steps = 12 * 10  # Forecast the final ten years, given previous data
    co2_by_month_training_data = co2_by_month[:-num_forecast_steps]
    co2_by_month_testing_data = co2_by_month[-num_forecast_steps:]

    trend_all = np.linspace(0., 1., len(co2_by_month)).reshape(-1, 1)
    trend_all = trend_all.astype(np.float32)
    trend = trend_all[:-num_forecast_steps, :]

    seasonality_all = pd.get_dummies(
        co2_by_month.index.month).values.astype(np.float32)
    seasonality = seasonality_all[:-num_forecast_steps, :]

    seasonality = torch.tensor(seasonality, dtype=torch.float64).float()
    trend = torch.tensor(trend, dtype=torch.float64).float()
    co2 = torch.tensor(co2_by_month_training_data.values.flatten(), dtype=torch.float64).float()

    n_changepoints = 12
    n_tp = seasonality.shape[0]
    t = torch.linspace(0, 1, n_tp, dtype=torch.float64)
    s = torch.linspace(0, max(t), n_changepoints + 2, dtype=torch.float64)[1:-1]
    A = torch.tensor((t[:, None] > s), dtype=torch.float64)

    X_pred = gen_fourier_basis(
        torch.where(seasonality)[1],
        p=seasonality.shape[-1],
        n=6
        )

    n_pred = X_pred.shape[-1]

    mean_CO2 = torch.tensor(co2_by_month_training_data.reset_index()['CO2'].mean())

    kernel = NUTS(gam)
    mcmc_gam = MCMC(kernel, 800, 300, num_chains=4)
    mcmc_gam.run(X_pred, mean_CO2, A, s, t, co2)

    print(mcmc_gam.summary())



if __name__ == '__main__':
    main()