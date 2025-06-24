import numpy as np
import matplotlib.pyplot as plt
from higher_order_ML_package.volterra_cpd import VolterraCPD, ConstrainedVolterraCPD
from higher_order_ML_package.utils import plot_graph,get_unique_interactions,gen_ground_truth_interactions,compute_MSE_metric

def generate_synthetic_data(N, T, HOI, SNR=None):
    X = np.random.randn(N-1, T)
    target_node = list(HOI.keys())[0]
    y = np.zeros(T)
    for interaction in HOI[target_node]:
        interaction = np.array(interaction) - 1 # Adjust for 0-based indexing
        y += np.prod(X[interaction, :], axis=0)

    if SNR is not None and SNR != np.inf:
        sigma = np.sqrt((1 / ((N-1) * T)) * np.sum(X**2) / (10**(SNR / 10)))
        X += np.random.normal(0, sigma, (N-1, T))

    return X.T, y


def plot_cost(cost, title):
    plt.figure()
    plt.plot(cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.semilogy()
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # ----- Shared configuration -----
    params = {
        "R": 3,
        "D": 3,
        "M": 1,
        "runs": 20,
        "reg_lambda": 0.001,
        "max_iter": 50,
        "tol": 1e-4,
        "max_inner_iter": 10,
        "verbose": True,
    }

    # ----- Generate synthetic data -----
    target_node = 0
    N = 9
    T = 5000
    HOI = {target_node: [[2, 3], [4, 5, 6], [7]]}
    SNR = 5
    X, y = generate_synthetic_data(N=N, T=T, HOI=HOI, SNR=SNR)

    # ----- VolterraCPD -----
    print("=== VolterraCPD ===")
    volterra_model = VolterraCPD(
        R=params["R"],
        D=params["D"],
        runs=params["runs"],
        reg_lambda=params["reg_lambda"],
        max_iter=params["max_iter"],
        tol=params["tol"],
        verbose=params["verbose"]
    )
    volterra_model.fit(X, y)
    print("Final cost:", volterra_model.cost_[-1])
    plot_cost(volterra_model.cost_, "VolterraCPD Training Cost")
    interactions_volterra = get_unique_interactions(volterra_model.weights_,params["D"])
    plot_graph({0:interactions_volterra}, N, weight_threshold=0.1, layout_quality=1.0, scale=3)
    ground_truth_interactions = gen_ground_truth_interactions(HOI,params["D"],N)
    plot_graph({0:ground_truth_interactions}, N, weight_threshold=0.1, layout_quality=1.0, scale=3)
    print('MSE error',compute_MSE_metric({0:interactions_volterra},{0:ground_truth_interactions})[target_node])

    # ----- ConstrainedVolterraCPD: ALS-LR -----
    # Adapt data X to the expected shape for ConstrainedVolterraCPD, with more features or time lags
    X_reshaped = np.zeros( (X.shape[0], X.shape[1]*(params["M"] + 1)) )
    for i in range(X.shape[1]):
        for j in range(params["M"] + 1):
            X_reshaped[:, i*(params["M"]+1) + j] = np.roll(X[:, i], j)

    X_reshaped = X_reshaped[params["M"]:,:]  # Remove the first M rows, because they are not valid for the M lags
    y = y[params["M"]:]  # Adjust y accordingly

    print("=== ConstrainedVolterraCPD (ALS-LR) ===")
    constrained_lr_model = ConstrainedVolterraCPD(
        R=params["R"],
        D=params["D"],
        M=params["M"],
        algorithm="ALS-LR",
        runs=params["runs"],
        reg_lambda=params["reg_lambda"],
        max_iter=params["max_iter"],
        tol=params["tol"],
        max_inner_iter=params["max_inner_iter"],
        verbose=params["verbose"]
    )
    constrained_lr_model.fit(X_reshaped, y)
    print("Final cost:", constrained_lr_model.cost_[-1])
    plot_cost(constrained_lr_model.cost_, "ConstrainedVolterraCPD (ALS-LR) Training Cost")
    interactions_volterra_lr = get_unique_interactions(constrained_lr_model.weights_,params["D"])
    plot_graph({0:interactions_volterra_lr}, N,weight_threshold=0.1, layout_quality=1.0, scale=3)

    # ----- ConstrainedVolterraCPD: ALS-SVD -----
    print("=== ConstrainedVolterraCPD (ALS-SVD) ===")
    constrained_svd_model = ConstrainedVolterraCPD(
        R=params["R"],
        D=params["D"],
        M=params["M"],
        algorithm="ALS-SVD",
        runs=params["runs"],
        reg_lambda=params["reg_lambda"],
        max_iter=params["max_iter"],
        tol=params["tol"],
        verbose=params["verbose"]
    )
    constrained_svd_model.fit(X_reshaped, y)
    print("Final cost:", constrained_svd_model.cost_[-1])
    plot_cost(constrained_svd_model.cost_, "ConstrainedVolterraCPD (ALS-SVD) Training Cost")
    interactions_volterra_svd = get_unique_interactions(constrained_svd_model.weights_,params["D"])
    plot_graph({0:interactions_volterra_svd}, N,weight_threshold=0.1, layout_quality=1.0, scale=3)
