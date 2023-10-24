import gpytorch
import matplotlib.pyplot as plt
import torch


def visualize_gp_belief_and_policy(model, likelihood, data, policy=None, next_x=None):

    with torch.no_grad():
        predictive_distribution = likelihood(model(data.xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region() 
        
        if policy is not None:
            acquisition_score = policy(data.xs.unsqueeze(1))

        if policy is None:
            plt.figure(figsize=(8, 3))
            plt.plot(data.xs, data.ys, label="objective", c="r")
            plt.scatter(data.train_x, data.train_y, marker="x", c="k", label="observations")
            plt.plot(data.xs, predictive_mean, label="mean")
            plt.fill_between(
                data.xs.flatten(),
                predictive_upper,
                predictive_lower,
                alpha=0.3,
                label="95% CI",
            )
            plt.legend()
            plt.show()
        else:
            fig, ax = plt.subplots(
                2, 1, 
                figsize=(8, 4), 
                sharex=True, 
                gridspec_kw={"height_ratios": [2, 1]}
            )
            ax[0].plot(data.xs, data.ys, label="objective", c="r")
            ax[0].scatter(data.train_x, data.train_y, marker="x", c="k", label="observations")
            ax[0].plot(data.xs, predictive_mean, label="mean")
            ax[0].fill_between(
                data.xs.flatten(),
                predictive_upper,
                predictive_lower,
                alpha=0.3,
                label="95% CI",
            )
            ax[0].set_ylabel("objective")

            if next_x is not None:
                ax[0].axvline(next_x, linestyle="dotted", c="k")

            ax[1].plot(data.xs, acquisition_score, c="g")
            ax[1].fill_between(
                data.xs.flatten(),
                acquisition_score,
                0,
                alpha=0.5
            )

            if next_x is not None:
                ax[1].axvline(next_x, linestyle="dotted", c="k")
                ax[1].set_ylabel("acquisition score")
            
            plt.show()
