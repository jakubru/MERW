import numpy as np
import math
import power_iteration

def compute_neighbours(A):
    neighbours_indices = []
    for row in range(len(A)):
        neighbours = []
        for col in range(len(A[row])):
            if _is_neighbour(A, row, col):
                neighbours.append(col)
        neighbours_indices.append(neighbours)
    return neighbours_indices

def compute_const(neighbours_counts, r, C):
    ret = []
    for i in range(r):
        consts = []
        for j in range(r):
            consts.append( C / (neighbours_counts[i] * neighbours_counts[j]))
        ret.append(consts)
    return ret

def compute_merw_consts(eigenvalue, eigenvector, r, alfa=0.5):
    ret = []
    for i in range(r):
        consts = []
        for j in range (r):
            consts.append(alfa*math.pow(eigenvalue, -2)*eigenvector[i]*eigenvector[j])
        ret.append(consts)
    return ret

def sim_rank(A, neighbours_counts, C=0.8, iterations=6):
    neighbours_indices = compute_neighbours(A)
    scores = np.identity(np.shape(A)[0])
    consts = compute_const(neighbours_counts, len(A), C)
    for i in range(iterations):
        old_scores = scores.copy()
        for j in range(len(A)):
            for k in range(len(A[j])):
                if j != k:
                    const = consts[j][k]
                    tmp_score = 0
                    for k1 in neighbours_indices[j]:
                        for j1 in neighbours_indices[k]:
                            tmp_score += old_scores[j1][k1]
                    scores[j][k] = const * tmp_score
    return scores


def merw_simrank(A, iterations=6):
    neighbours_indices = compute_neighbours(A)
    scores = np.identity(np.shape(A)[0])
    eigenvalue, eigenvector = power_iteration(A)
    consts = compute_merw_consts(eigenvalue, eigenvector, len(A))
    for i in range(iterations):
        old_scores = scores.copy()
        for j in range(len(A)):
            for k in range(len(A[j])):
                if j != k:
                    const = consts[j][k]
                    tmp_score = 0
                    for k1 in neighbours_indices[j]:
                        for j1 in neighbours_indices[k]:
                            tmp_score += old_scores[j1][k1]/eigenvector[j1][k1]
                    scores[j][k] = const * tmp_score
    return scores

def _is_neighbour(A, a, b):
    return A[a][b]
