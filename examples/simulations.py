import numpy as np
import matplotlib.pyplot as plt
import time, os, pickle
import pandas as pd
from datetime import datetime
from higher_order_ML_package.volterra_cpd import VolterraCPD, ConstrainedVolterraCPD
from higher_order_ML_package.utils import compute_MSE_metric, get_unique_interactions, gen_ground_truth_interactions,reorder_all_weights
from scipy.special import comb

def generate_synthetic_data(N, T, HOI, SNR=None):
    N=N-1
    X = np.random.randn(N, T)
    target_node = list(HOI.keys())[0]
    y = np.zeros(T)
    for interaction in HOI[target_node]:
        interaction = np.array(interaction) - 1
        y += np.prod(X[interaction, :], axis=0)
    if SNR is not None and SNR != np.inf:
        sigma = np.sqrt((1 / (N * T)) * np.sum(X**2) / (10**(SNR / 10)))
        X += np.random.normal(0, sigma, (N, T))
    return X.T, y

def reshape_for_constrained(X, y, M):
    X_new = np.zeros((X.shape[0], X.shape[1] * (M + 1)))
    for i in range(X.shape[1]):
        for j in range(M + 1):
            X_new[:, i * (M + 1) + j] = np.roll(X[:, i], j)
    return X_new[M:],y[M:]

def run_experiment(SNR_vals, T_vals, params, N, HOI, C, num_repeats=5):
    target_node = list(HOI.keys())[0]
    ground_truth = gen_ground_truth_interactions(HOI, params["D"], N, C=C)
    records = []
    for snr in SNR_vals:
        for T in T_vals:
            for repeat in range(num_repeats):
                X, y = generate_synthetic_data(N=N, T=T, HOI=HOI, SNR=snr)
                X_constr, y = reshape_for_constrained(X, y, params["M"])

                for method in ["ALS-SVD", "ALS-LR", "ALS"]:
                    start = time.time()
                    if method == "ALS-SVD":
                        model = ConstrainedVolterraCPD(
                            R=params["R"], D=params["D"], M=params["M"], reg_lambda=params["reg_lambda"],
                            algorithm="ALS-SVD",runs=params["runs"], max_iter=params["max_iter"], tol=params["tol"],
                            verbose=False
                        )
                        model.fit(X_constr, y)
                        duration = time.time() - start
                        weights_reordered = reorder_all_weights(model.all_weights_, C)
                        interactions = get_unique_interactions(weights_reordered, params["D"])

                    elif method == "ALS-LR":
                        model = ConstrainedVolterraCPD(
                            R=params["R"], D=params["D"], M=params["M"],reg_lambda=params["reg_lambda"],  algorithm="ALS-LR",
                            runs=params["runs"], max_iter=params["max_iter"], tol=params["tol"],
                            max_inner_iter=params["max_inner_iter"],
                            inner_tol=params["inner_tol"],
                            verbose=False
                        )
                        model.fit(X_constr, y)
                        duration = time.time() - start
                        weights_reordered = reorder_all_weights(model.all_weights_, C)
                        interactions = get_unique_interactions(weights_reordered, params["D"])

                    elif method == "ALS":
                        model = VolterraCPD(
                            R=params["R"], D=params["D"], reg_lambda=params["reg_lambda"],
                            runs=params["runs"], max_iter=params["max_iter"], tol=params["tol"],
                            verbose=False
                        )
                        model.fit(X_constr, y)
                        duration = time.time() - start
                        weights_reordered = reorder_all_weights(model.weights_, C, ALS=True)
                        interactions = get_unique_interactions(weights_reordered, params["D"])

                    mse = compute_MSE_metric({target_node: interactions}, {target_node: ground_truth})[target_node]

                    records.append({
                        "SNR": snr,
                        "T": T,
                        "method": method,
                        "repeat": repeat,
                        "time_sec": duration,
                        "MSE": mse,
                        "interactions": interactions  # Optional: can be removed for compactness
                    })

                    print(f"T={T}, SNR={snr}, {method}, run {repeat + 1}/{num_repeats}: MSE={mse:.2e}, time={duration:.2f}s")

    return pd.DataFrame(records)


def create_simulation_dir(simulation_id=None):
    base_dir = "simulation_results"
    os.makedirs(base_dir, exist_ok=True)

    if simulation_id is None:
        simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_dir = os.path.join(base_dir, f"simulation_{simulation_id}")
    os.makedirs(sim_dir, exist_ok=True)
    return sim_dir, simulation_id


def load_simulation(sim_id, base_dir="simulation_results"):
    sim_dir = os.path.join(base_dir, sim_id)

    # Load results
    with open(os.path.join(sim_dir, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    # Load timing data
    timing_df = pd.read_csv(os.path.join(sim_dir, "timing.csv"))

    return results, timing_df


def plot_timing(results_df, T_vals, save_dir=None):
    methods = results_df["method"].unique()
    colors = {"ALS-SVD": "tab:blue", "ALS-LR": "tab:orange", "ALS": "tab:green"}
    linestyles = {"ALS-SVD": "-", "ALS-LR": "--", "ALS": ":"}

    for snr, df_snr in results_df.groupby("SNR"):
        plt.figure(figsize=(6, 4))
        for i, method in enumerate(methods):
            df_m = df_snr[df_snr["method"] == method]
            means = df_m.groupby("T")["time_sec"].mean()
            stds = df_m.groupby("T")["time_sec"].std()

            avg_T_gap = np.mean(np.diff(T_vals))
            jitter_frac = 0.05  # 5% of the T-gap
            max_offset = avg_T_gap * jitter_frac

            # Compute symmetric jitter offsets centered around 0
            if len(methods) == 1:
                offsets = [0]
            else:
                offsets = np.linspace(-max_offset, max_offset, len(methods))

            T_vals_jittered = np.array(T_vals) + offsets[i]

            plt.errorbar(T_vals_jittered, means, yerr=stds, label=method,
                         color=colors.get(method, None), linestyle=linestyles.get(method, None), marker='o')

        plt.title(f"Runtime vs T (SNR={snr})")
        plt.xlabel("T")
        plt.ylabel("Time (s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"timing_plot_snr_{snr}.png"))
        plt.show()

def plot_results(results_df, T_vals, save_dir=None):
    methods = results_df["method"].unique()
    colors = {"ALS-SVD": "tab:blue", "ALS-LR": "tab:orange", "ALS": "tab:green"}
    linestyles = {"ALS-SVD": "-", "ALS-LR": "--", "ALS": ":"}

    for snr, df_snr in results_df.groupby("SNR"):
        plt.figure(figsize=(6, 4))
        for i, method in enumerate(methods):
            df_m = df_snr[df_snr["method"] == method]
            means = df_m.groupby("T")["MSE"].mean()
            stds = df_m.groupby("T")["MSE"].std()

            avg_T_gap = np.mean(np.diff(T_vals))
            jitter_frac = 0.05  # 5% of the T-gap
            max_offset = avg_T_gap * jitter_frac

            # Compute symmetric jitter offsets centered around 0
            if len(methods) == 1:
                offsets = [0]
            else:
                offsets = np.linspace(-max_offset, max_offset, len(methods))

            T_vals_jittered = np.array(T_vals) + offsets[i]

            plt.errorbar(T_vals_jittered, means, yerr=stds, label=method,
                         color=colors.get(method, None), linestyle=linestyles.get(method, None), marker='o')

        plt.title(f"MSE vs T (SNR={snr})")
        plt.xlabel("T")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"mse_plot_snr_{snr}.png"))
        plt.show()

if __name__ == "__main__":
    params = {
        "R": 3,
        "D": 3,
        "M": 7,
        "reg_lambda": 0.0001,
        "runs": 10,
        "max_iter": 50,
        "tol": 1e-4,
        "max_inner_iter": 2,
        "inner_tol": 1e-2,
        "verbose": False,
    }

    T_vals = [1000, 3000, 5000, 7000, 9000, 11000]
    SNR_vals = [np.inf, 3]
    N = 9
    HOI = {0: [[2, 3], [4, 5, 6], [7]]}

    C = np.zeros((N,params['M']+1))
    C[0,:] = np.ones(params['M']+1) * (1/(params['M']+1))
    C[1:,0] = 1

    #sim_ID = 'simulation_20250514_140002'
    #sim_ID = 'simulation_20250527_161812'

    sim_ID = None

    if not sim_ID:
        # ==== RUN SIMULATION
        print("=== Running Simulation ===")
        indep_model_param_naive = int(comb(N * (params['M'] + 1) + params['D'] - 1, params['D']))
        indep_model_param_efficient = int(comb(N + params['D'] - 1, params['D'] ) + params['D']*N*params['M'])
        print('Number of independent parameters naive case:',indep_model_param_naive)
        print('Number of independent parameters efficient case:',indep_model_param_efficient)

        # results, timing_df = run_experiment(SNR_vals, T_vals, params, N, HOI, C)
        results = run_experiment(SNR_vals, T_vals, params, N, HOI, C, num_repeats=10)

        sim_dir, sim_id = create_simulation_dir()

        # Save results
        with open(os.path.join(sim_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f)

        results.to_csv(os.path.join(sim_dir, "timing.csv"), index=False)

        plot_results(results, T_vals, save_dir=sim_dir)
        plot_timing(results, T_vals, save_dir=sim_dir)

        print(f"\n✅ Simulation {sim_id} completed and saved to {sim_dir}")
    else:
        results, timing_df = load_simulation(sim_ID)
        sim_dir = os.path.join('simulation_results', sim_ID)
        print(f"\n✅ Loaded simulation {sim_ID} from {sim_dir}")
        plot_results(results, T_vals, save_dir=sim_dir)
        plot_timing(results, T_vals, save_dir=sim_dir)