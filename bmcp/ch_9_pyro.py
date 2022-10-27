import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pyro
import torch
import arviz as az
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


def main():

    msn_arrivals = load_data()

    ## Prior Predictive Check ##
    normal_prior = Predictive(normal, {}, num_samples=100)(None)
    gumbel_prior = Predictive(gumbel, {}, num_samples=100)(None)

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
    
    normal_posterior = Predictive(normal, normal_posterior_samples)(None)
    gumbel_posterior = Predictive(gumbel, gumbel_posterior_samples)(None)

    arviz_normal = az.from_pyro(
        posterior=normal_mcmc,
        prior=normal_prior,
        posterior_predictive=normal_posterior
    )

    arviz_gumbel = az.from_pyro(
        posterior=gumbel_mcmc,
        prior=gumbel_prior,
        posterior_predictive=gumbel_posterior
    )

    # Inference Diagnostics ##
    az.plot_trace(arviz_normal, kind='rank_bars')
    az.plot_posterior(arviz_normal)
    #az.plot_ppc(arviz_normal, observed=True, num_pp_samples=20) # does not work
    plt.tight_layout()
    plt.show()

    ## Model Selection ##
    # compare_dict = {'normal': arviz_normal, 'gumbel': arviz_gumbel}
    # print(az.compare(compare_dict, ic='loo'))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.kdeplot(msn_arrivals, ax=ax[0], color='black', label='Obs. data')
    sns.kdeplot(normal_posterior['obs'], ax=ax[0], color='blue', label='Post. pred. mean')
    plt.legend()
    sns.kdeplot(msn_arrivals, ax=ax[1], color='black', label='Obs. data')
    sns.kdeplot(gumbel_posterior['obs'], ax=ax[1], color='blue', label='Post. pred. mean')
    plt.legend()
    plt.suptitle('Posterior Predictive Check')
    plt.show()


if __name__ == "__main__":
    main()