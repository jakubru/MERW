import numpy as np
import numba
import power_iteration


def general_graph_laplacian(A):
    P = power_iteration.compute_transition_matrix(A)
    PI = np.diag(power_iteration.compute_stationary_distribution(P))
    I = np.identity(len(P))
    return np.dot(PI, I - P)


def me_combinatorial_graph_laplacian(A):
    ev, v = power_iteration.power_iteration(A)
    D_v = np.diag(v)
    return np.power(D_v, 2) - np.dot(D_v, np.dot(A, D_v)) / ev


def sym_norm_me_graph_laplacian(A):
    ev, v = power_iteration.power_iteration(A)
    I = np.identity(len(A))
    return I - A / ev


@numba.jit('float64[:,:](float64[:,:], float64[:])')
def hitting_time(L, v):
    L_ = np.linalg.pinv(L)
    H = np.zeros(np.shape(L_))
    for i in range(len(L_)):
        for j in range(len(L_[i])):
            h = []
            for k, _ in enumerate(L_):
                h.append((L_[i, k] + - L_[i, j] - L_[j, k] + L_[j, j]) * v[k])
            H[i, j] = np.sum(h)
    return H


@numba.jit('float64[:,:](float64[:,:])')
def commute_time(L):
    L_ = np.linalg.pinv(L)
    C = np.zeros(np.shape(L_))
    for i in range(len(L_)):
        for j in range(len(L_[i])):
            C[i, j] = L_[i, i] + L_[j, j] - 2 * L_[i, j]
    return C


def commute_kernel(L):
    return np.linalg.pinv(L)
