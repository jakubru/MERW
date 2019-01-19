import numpy as np
import scipy

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


def power_iteration(A):
    n, d = A.shape
    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)
    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)
        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break
        v = v_new
        ev = ev_new
    return ev_new, v_new


def compute_neighbours(A):
    neighbours_indices = []
    for row in range(len(A)):
        neighbours = []
        for col in range(len(A[row])):
            if _is_neighbour(A, row, col):
                neighbours.append(col)
        neighbours_indices.append(neighbours)
    return neighbours_indices


def _is_neighbour(A, a, b):
    return A[a][b]



def compute_transition_matrix(A):
    P = np.zeros(np.shape(A))
    i = 0
    for element in A:
        sum_ = sum(element)
        for j in range(len(element)):
            if(A[i,j] == 1):
                P[i,j] = 1/float(sum_)
        i+=1
    return P

def compute_stationary_distribution(P):
    ev,v = power_iteration(P)
    return v/(sum(v))