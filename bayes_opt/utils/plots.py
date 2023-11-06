import matplotlib.pyplot as plt
import torch


def visualize_progress_and_policy(data, policy, next_x=None, extent=[0, 2, 0, 2]):

    with torch.no_grad():
        acquisition_score = policy(data.xs.unsqueeze(1)).reshape(101, 101).T

    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    c = ax[0].imshow(data.ys.reshape(101, 101).T, origin="lower", extent=extent)
    ax[0].set_xlabel(r"$C$")
    ax[0].set_ylabel(r"$\gamma$")
    plt.colorbar(c, ax=ax[0])

    ax[0].scatter(
        data.train_x[..., 0], 
        data.train_x[..., 1], 
        marker="x", 
        c="k"
    )

    c = ax[1].imshow(acquisition_score, origin="lower", extent=extent)
    ax[1].set_xlabel(r"$C$")
    ax[1].set_ylabel(r"$\gamma$")
    plt.colorbar(c, ax=ax[1])

    if next_x is not None:
        ax[1].scatter(
            next_x[..., 0],
            next_x[..., 1],
            c="r",
            marker="*",
            s=250,
            label="next query"
        )
    
    plt.tight_layout()

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


def visualize_experiment(result, n_queries, n_experiments, label=None):

    def ci(y):
        return 2 * y.std(axis=0) / torch.sqrt(torch.tensor(n_experiments))

    mean_incumbent = result.mean(axis=0)[:, 0]
    ci_incumbent = ci(result)[:, 0]

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(torch.arange(n_queries), mean_incumbent, label="Mean")
    ax.fill_between(
        torch.arange(n_queries),
        mean_incumbent - ci_incumbent,
        mean_incumbent + ci_incumbent,
        alpha=0.2,
        label="CI",
    )
    ax.grid(True)

    return fig, ax