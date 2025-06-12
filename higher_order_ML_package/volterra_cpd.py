# mymlpackage/volterra_cpd.py
from sklearn.base import BaseEstimator
import numpy as np
from .estimation import ALS, ALS_SVD, ALS_LR

from multiprocessing import Pool,cpu_count
import cProfile
import pstats
import io

def profile_func(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumtime'  # can be 'tottime' or 'calls'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper


class VolterraCPD(BaseEstimator):
    def __init__(self, R=3, D=3, reg_lambda=0.001, max_iter=100, runs=1, tol=1e-4, verbose=False):
        self.R = R
        self.D = D
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.runs = runs
        self.tol = tol
        self.verbose = verbose

        self.weights_ = None
        self.cost_ = None

    def fit(self, X, y):
        args = [(X, y, self.R, self.D, self.reg_lambda, self.max_iter, self.tol, self.verbose) for _ in range(self.runs)]

        if self.runs > 1:
            with Pool(processes=min(self.runs, cpu_count())) as pool:
                results = pool.starmap(ALS, args)

            # Choose the best result with the lowest error
            best_res = min(results, key=lambda res: res[1][-1])
        else:
            # Single run case
            best_res = ALS(*args[0])

        self.weights_ = best_res[0]
        self.cost_ = best_res[1]
        return self

    def predict(self, X_new):
        feature_mapped_data = np.vstack([(1 / (self.M + 1)) * np.ones((self.M + 1, X_new.shape[0])), X_new.T])
        prod = 1
        for d in range(self.D):
            prod *= self.weights_[d].T @ feature_mapped_data
        reconstructed_signal = np.ones(self.R) @ prod
        return reconstructed_signal



class ConstrainedVolterraCPD(BaseEstimator):
    def __init__(self, R=3, D=3, M=1, reg_lambda=0.001, max_iter=100, runs=1, tol=1e-4, max_inner_iter=2,inner_tol=1e-2, algorithm='ALS-LR', verbose=False):
        self.R = R
        self.D = D
        self.M = M
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.runs = runs
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.algorithm = algorithm  # 'low_rank_ALS', 'ALS-SVD'
        self.verbose = verbose

        self.all_weights_ = None
        self.weights_ = None
        self.coefficients_ = None
        self.cost_ = None

    def fit(self, X, y):

        if self.algorithm == 'ALS-LR':
            # Use the ALS algorithm
            fit_func = ALS_LR
            args = [(X, y, self.R, self.D, self.M, self.reg_lambda, self.max_iter, self.tol, self.max_inner_iter, self.inner_tol, self.verbose) for _ in range(self.runs)]
        elif self.algorithm == 'ALS-SVD':
            # Use the ALS-SVD algorithm
            fit_func = ALS_SVD
            args = [(X, y, self.R, self.D, self.M, self.reg_lambda, self.max_iter, self.tol, self.verbose) for _ in range(self.runs)]
        else:
            raise ValueError("Algorithm not recognized. Use 'ALS-LR' or 'ALS-SVD'.")

        if self.runs > 1:
            with Pool(processes=min(self.runs, cpu_count())) as pool:
                results = pool.starmap(fit_func, args)

            # Choose the best result with the lowest error
            best_res = min(results, key=lambda res: res[3][-1])
        else:
            # Single run case
            best_res = fit_func(*args[0])

        self.all_weights_ = best_res[0]
        self.weights_ = best_res[1]
        self.coefficients_ = best_res[2]
        self.cost_ = best_res[3]
        return self

    def predict(self, X_new):
        feature_mapped_data = np.vstack([(1 / (self.M + 1)) * np.ones((self.M + 1, X_new.shape[0])), X_new.T])
        prod = 1
        for d in range(self.D):
            prod *= self.all_weights_[d].T @ feature_mapped_data
        reconstructed_signal = np.ones(self.R) @ prod
        return reconstructed_signal
