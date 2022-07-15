import pandas as pd





def summary(samples):
    """utility function to print latent sites' quantile information"""

    site_stats = {}

    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(
            percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[
            ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
            
    return site_stats