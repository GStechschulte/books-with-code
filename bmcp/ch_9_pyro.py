import enum
from matplotlib import axes
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import logging
import pyro
import torch
import arviz as az
#import torch.distributions as dist
import pyro.distributions as dist
from pyro.infer import Predictive, NUTS, MCMC



def load_data():
    df = pd.read_csv('./data/948363589_T_ONTIME_MARKETING.zip', low_memory=False)
    #flights = pl.read_csv('./data/948363589_T_ONTIME_MARKETING.csv', ignore_errors=True)

    # print((flights.filter(
    #     (pl.col('DESC') == 'MSN') &
    #     (pl.col('ORIGIN') == 'MSP') |
    #     (pl.col('ORIGIN') == 'DTW')
    # )))

    msn_arrivals = df[(df["DEST"] == 'MSN') & df["ORIGIN"]
                  .isin(["MSP", "DTW"])]["ARR_DELAY"]
    
    return torch.from_numpy(msn_arrivals.values)


def normal(obs=None):

    loc = pyro.sample('loc', dist.Normal(0., 30.))
    scale = pyro.sample('scale', dist.HalfNormal(5.))
    
    pyro.sample(
            'obs', dist.Normal(loc, scale),
            obs=obs
        )


def gumbel(obs=None):

    loc = pyro.sample('loc', dist.Normal(0., 20.))
    beta = pyro.sample('beta', dist.HalfNormal(5.))
    
    pyro.sample(
            'obs', dist.Gumbel(loc, beta),
            obs=obs
        )


def predictive_check(model_1, model_2, obs):

    model_1_pred = Predictive(model_1, {}, num_samples=100)(None)['obs']
    model_2_pred = Predictive(model_2, {}, num_samples=100)(None)['obs']

    sns.histplot(model_1_pred.flatten(), kde=True)
    plt.show()

    # for i, (pred_1, pred_2) in enumerate(zip(model_1_pred, model_2_pred)):
    # sns.histplot(model_1_pred.flatten(), kde=True)
    # plt.show()


def main():

    msn_arrivals = load_data()

    ## Prior Predictive Check ##
    #predictive_check(normal, gumbel, msn_arrivals)

    ## Inference ##
    normal_mcmc = MCMC(
        NUTS(normal, adapt_step_size=True),
        500, 100, num_chains=4
        )
    normal_mcmc.run(msn_arrivals)
    
    gumbel_mcmc = MCMC(
        NUTS(gumbel, adapt_step_size=True),
         500, 100, num_chains=4)
    gumbel_mcmc.run(msn_arrivals)

    ## Posterior Predictive Check ##
    normal_posterior_samples = normal_mcmc.get_samples(1000)
    gumbel_posterior_samples = gumbel_mcmc.get_samples(1000)
    
    normal_predictive = Predictive(normal, normal_posterior_samples)(None)['obs']
    gumbel_predictive = Predictive(gumbel, gumbel_posterior_samples)(None)['obs']

    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.kdeplot(msn_arrivals, ax=ax[0], color='black', label='Obs. data')
    sns.kdeplot(normal_predictive, ax=ax[0], color='blue', label='Post. pred. mean')
    plt.legend()
    sns.kdeplot(msn_arrivals, ax=ax[1], color='black', label='Obs. data')
    sns.kdeplot(gumbel_predictive, ax=ax[1], color='blue', label='Post. pred. mean')
    plt.legend()
    plt.suptitle('Posterior Predictive Check')
    plt.show()

    # arviz_normal = az.from_pyro(
    #     posterior=normal_mcmc,
    #     posterior_predictive=normal_predictive
    # )

    # arviz_gumbel = az.from_pyro(
    #     posterior=gumbel_mcmc,
    #     posterior_predictive=gumbel_predictive
    # )

    ## Inference Diagnostics ##
    # az.plot_trace(arviz_normal)
    # az.plot_rank(arviz_normal)
    # az.plot_posterior(arviz_normal)
    # az.plot_ppc(arviz_normal, observed=True, num_pp_samples=20)
    # plt.show()

    ## Model Selection ##
    # compare_dict = {'normal': arviz_normal, 'gumbel': arviz_gumbel}
    # print(az.compare(compare_dict, ic='loo'))


if __name__ == "__main__":
    main()