import numpy as np
from scipy.linalg import khatri_rao
from .utils import cost_fn_efficient

def ALS(X, y, R, D, reg_lambda=None, max_iter = None, tol=None, verbose=False):

    X_feature_mapped = np.vstack([np.ones((1, len(y))), X.T])
    feature_map_dimension = X_feature_mapped.shape[0]

    # Initialize factor matrices randomly
    W = [np.random.randn(feature_map_dimension,R) for _ in range(D)]
    cost = []

    H_d = 1
    G_product = 1
    for d in range(D):
        W[d] /= np.linalg.norm(W[d], 'fro')
        H_d *= W[d].copy().T@W[d].copy()
        G_product *= W[d].T @ X_feature_mapped

    # zigzag = list(range(D)) + list(range(D - 2, 0, -1))
    # Iterate until convergence or max_iter is reached
    for iteration in range(max_iter):
        for d in range(D):
            H_d /= W[d].T@W[d]  # Remove the current W[d] from the H_d matrix.
            G_product /= W[d].T @ X_feature_mapped
            G = khatri_rao(G_product, X_feature_mapped)

            # Solve least squares for W^(d) and apply reg
            vec_W_d = np.linalg.solve(G @ G.T + reg_lambda * np.kron(H_d, np.eye(feature_map_dimension)), G @ y)
            W[d] = vec_W_d.copy().reshape((feature_map_dimension,R),order='F')

            # Normalize W[d] except for the last one
            if d!=D-1:
                W[d] /= np.linalg.norm(W[d], 'fro')

            H_d *= W[d].T@W[d]
            G_product *= W[d].T @ X_feature_mapped

        # Save cost
        cost.append(cost_fn_efficient(G_product, H_d, y, reg_lambda=reg_lambda))

        # Check for convergence
        if iteration > 0:
            cost_difference = cost[-2] - cost[-1]
            if cost_difference < tol:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations.")
                break
    return W, cost


def ALS_SVD(X, y, R, D, M, reg_lambda, max_iter, tol, verbose=False):
    N = int(X.shape[1] / (M+1)) + 1
    T = len(y)
    feature_map_dimension = N * (1 + M)
    feature_mapped_data = np.vstack([(1 / (M + 1)) * np.ones((M + 1, T)), X.T])

    cost = []
    coefficients = np.zeros((N-1, M+1))
    W = []
    for d in range(D):
       W.append(np.zeros((N, R)))

    # Initialize factor matrices randomly
    U = [np.random.randn(feature_map_dimension,R) for _ in range(D)]
    U_alt = [np.zeros((1+(N-1)*(M+1),R)) for _ in range(D)]

    # U = []
    # for d in range(D):
    #     U.append(np.zeros((feature_map_dimension, R)))
    #     U[d][::M+1, :] = np.random.randn(len(U[d][::M+1]), R)

    H_d = 1
    G_product = 1
    for d in range(D):
        U[d] /= np.linalg.norm(U[d], 'fro')
        H_d *= U[d].copy().T@U[d].copy()  # np.linalg.norm(W[d], axis=0) ** 2
        G_product *= U[d].T @ feature_mapped_data

    # Iterate until convergence or max_iter is reached
    try:
        for iteration in range(max_iter):
            for d in range(D):
                H_d /= U[d].T@U[d]  # Remove the current W[d] from the H_d matrix.
                G_product /= U[d].T @ feature_mapped_data
                G = khatri_rao(G_product, feature_mapped_data)

                # Solve least squares for W^(d) and apply reg
                vec_U_d = np.linalg.lstsq(G @ G.T + reg_lambda * np.kron(H_d, np.eye(feature_map_dimension)), G @ y)[0]
                U[d] = vec_U_d.copy().reshape((feature_map_dimension,R),order='F')

                # Normalize W[d] except for the last one
                if d!=D-1:
                    U[d] /= np.linalg.norm(U[d], 'fro')

                H_d *= U[d].T@U[d]
                G_product *= U[d].T @ feature_mapped_data

            # SVD
            if iteration % 1 == 0:
                U_concat = np.concatenate([U[d_ind] for d_ind in range(D)],axis=1)
                U_concat[0:M+1, :] = np.ones((M+1,1)) @ np.mean(U_concat[0:M+1, :],axis=0,keepdims=True)  # means of the first M rows
                for d_ind in range(D):
                    W[d_ind][0,:] = np.mean(U_concat[0:M+1, d_ind*R:(d_ind+1)*R],axis=0,keepdims=True)
                for n in range(1,N):
                    U_block, S_block, Vt_block = np.linalg.svd(U_concat[n * (M+1): (n + 1) * (M+1), :], full_matrices=False)
                    U_concat[n * (M+1): (n + 1) * (M+1), :] = S_block[0] * np.outer(U_block[:,0], Vt_block[0, :])
                    # Store the C_tilde
                    coefficients[n-1,:] = U_block[:,0]
                    for d_ind in range(D):
                        W[d_ind][n, :] = S_block[0] * Vt_block[0,d_ind*R:(d_ind+1)*R]
                H_d=1
                G_product=1
                for d_ind in range(D):
                    U[d_ind] = U_concat[:,d_ind*R:(d_ind+1)*R]  # Update U[d] with the new W
                    H_d *= U[d_ind].copy().T @ U[d_ind].copy()
                    G_product *= U[d_ind].T @ feature_mapped_data

            # Save cost
            cost.append(cost_fn_efficient(G_product, H_d, y, reg_lambda=reg_lambda))


            # Check for convergence
            if iteration > 0:
                cost_diff = cost[-2] - cost[-1]
                if cost_diff < tol:
                    if verbose:
                        print(f"Converged in {iteration + 1} iterations.")
                    break

        for d in range(D):
            U_alt[d][0, :] = W[d][0, :]
            for n in range(N - 1):
                U_alt[d][1 + n * (M + 1): 1 + (n + 1) * (M + 1), :] = np.outer(coefficients[n,:], W[d][n+1, :])
    except:
        # Sometimes the SVD method fails to converge
        cost.append(np.inf)
    return U_alt, W, coefficients, cost


def ALS_LR(X, y, R, D, M, reg_lambda, max_iter, tol, max_inner_iter, verbose=False):
    N = int(X.shape[1] / (M+1)) + 1
    T = len(y)
    feature_map_dimension = N * (1 + M)
    feature_mapped_data = np.vstack([(1 / (M + 1)) * np.ones((M + 1, T)), X.T])

    cost = []

    # Initialize factor matrices and coefficient vectors randomly
    U = [np.random.randn(feature_map_dimension,R) for _ in range(D)]  # Just vectors
    U_alt = [np.zeros((1+ (N-1)*(M+1),R)) for _ in range(D)]  # Just vectors
    W = [np.random.randn(N,R) for _ in range(D)]
    C = [np.random.randn(feature_map_dimension) for _ in range(D)]  # Just vectors

    for d in range(D):
        C[d][:M+1] = 1
        #C[d][M+1::2] = 1  # used for testing
        W[d] /= np.linalg.norm(W[d], 'fro')

    P = np.ones((R, T))
    for index in range(D):
        vec = np.zeros((R,T))
        for n in range(N):
            vec += np.outer(W[index][n, :] , (C[index][n * (M + 1):(n + 1) * (M + 1)] @ feature_mapped_data[n * (M + 1):(n + 1) * (M + 1),:]) )
        P *= vec

    H = 1
    for index in range(D):
        UtU = 0
        for n in range(N):
            UtU += (C[index][n * (M + 1):(n + 1) * (M + 1)].T @ C[index][n * (M + 1):(n + 1) * (M + 1)]) * np.outer(
                W[index][n, :], W[index][n, :])
        H *= UtU

    # Iterate until convergence or max_iter is reached
    for iteration in range(max_iter):
        for d in range(D):
            vec_batch = np.zeros((R,T))
            for n in range(N):
                vec_batch += np.outer(W[d][n, :],(C[d][n * (M + 1):(n + 1) * (M + 1)] @ feature_mapped_data[n * (M + 1):(n + 1) * (M + 1),:]))
            P /= vec_batch

            UtU = 0
            for n in range(N):
                UtU += (C[d][n*(M+1):(n+1)*(M+1)].T @ C[d][n*(M+1):(n+1)*(M+1)]) * np.outer(W[d][n,:], W[d][n,:])
            H /= UtU

            for i in range(max_inner_iter):

                G = np.zeros((T,N*R))
                for n in range(N):
                    G[:, n*R:(n+1)*R] = (C[d][n*(M+1):(n+1)*(M+1)] @ feature_mapped_data[n*(M+1):(n+1)*(M+1), :])[:, np.newaxis] * P.T

                # Solve least squares for W^(d) and apply reg
                diag = np.diag(np.array([C[d][n*(M+1):(n+1)*(M+1)].T @ C[d][n*(M+1):(n+1)*(M+1)] for n in range(N)]))
                vec_W_d = np.linalg.lstsq(G.T @ G + reg_lambda * np.kron(diag, H), G.T @ y)[0]
                W[d] = vec_W_d.copy().reshape((N,R))  # No order f this time

                G = np.zeros((T, N * (M+1)))
                for n in range(N):
                    might_give_underflow = np.vecmat(W[d][n, :], P)[:, np.newaxis]
                    G[:, n*(M+1):(n+1)*(M+1)] = might_give_underflow * feature_mapped_data[n*(M+1):(n+1)*(M+1),:].T

                diag = np.diag(np.concatenate([np.full(M + 1, W[d][n].T @ H @ W[d][n]) for n in range(N)]))
                C[d][M+1:] = np.linalg.lstsq(G[:, M + 1:].T @ G[:, M + 1:] + reg_lambda * diag[M+1:, M+1:], G[:, M + 1:].T @ (y - G[:, :M + 1] @ C[d][:M+1]))[0]

                # Normalize blocks
                for n in range(1,N):
                    norm = np.linalg.norm(C[d][n*(M+1):(n+1)*(M+1)])
                    C[d][n*(M+1):(n+1)*(M+1)] /= norm
                    W[d][n,:] *= norm

            # Normalization on full updated factor matrix
            if d!=D-1:
                norm = 0
                for n in range(N):
                    # Compute the frob norm of U[d]
                    norm += np.linalg.norm(W[d][n, :])**2 * np.linalg.norm(C[d][n * (M + 1):(n + 1) * (M + 1)])**2
                W[d] /= np.sqrt(norm)

            vec = np.zeros((R,T))
            for n in range(N):
                vec += np.outer(W[d][n, :],(C[d][n * (M + 1):(n + 1) * (M + 1)] @ feature_mapped_data[n * (M + 1):(n + 1) * (M + 1),:]))
            P *= vec

            UtU = 0
            for n in range(N):
                UtU += (C[d][n*(M+1):(n+1)*(M+1)].T @ C[d][n*(M+1):(n+1)*(M+1)]) * np.outer(W[d][n,:], W[d][n,:])
            H *= UtU

        cost.append(cost_fn_efficient(P, H, y, reg_lambda=reg_lambda))

        # Check for convergence
        if iteration > 0:
            cost_diff = cost[-2] - cost[-1]
            if cost_diff < tol:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations.")
                break

        for d in range(D):
            for n in range(N):
                U[d][n * (M + 1):(n + 1) * (M + 1), :] = np.outer(C[d][n * (M + 1):(n + 1) * (M + 1)],W[d][n, :])

        for d in range(D):
            U_alt[d][0,:] = W[d][0, :]
            for n in range(N-1):
                U_alt[d][1 + n*(M+1): 1 + (n+1)*(M+1),:] = np.outer(C[d][(1+n)*(M+1):(n+2)*(M+1)],W[d][n+1,:])
    return U_alt, W, C, cost
