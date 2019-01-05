import numpy as np


def compute_neighbours(A):
    neighbours_indices = []
    for row in range(len(A)):
        neighbours = []
        for col in range(len(A[row])):
            if _is_neighbour(A, row, col):
                neighbours.append(col)
        neighbours_indices.append(neighbours)
    return neighbours_indices


def sim_rank(A, neighbours_counts, C=0.8, iterations=6):
    neighbours_indices = compute_neighbours(A)
    scores = np.identity(np.shape(A)[0])
    for i in range(iterations):
        old_scores = scores.copy()
        for j in range(len(A)):
            for k in range(len(A[j])):
                # if j != k:
                    const = C / (neighbours_counts[j] * neighbours_counts[k])
                    tmp_score = 0
                    for k1 in neighbours_indices[j]:
                        for j1 in neighbours_indices[k]:
                            tmp_score += old_scores[j1][k1]
                    scores[j][k] = const * tmp_score
                # else:
                #     scores[j][k] = 1
    return scores


def _is_neighbour(A, a, b):
    return A[a][b]
