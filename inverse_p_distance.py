from math import pow

import numpy as np

import power_iteration


def inverse_p_distance(A, alfa=0.7, depth=5):
    P = power_iteration.compute_transition_matrix(A)
    neigh_list = power_iteration.compute_neighbours(A)
    inverse_p_distances = list()
    for i in range(len(A)):
        distances = [[0.0 for _ in range(depth + 1)] for _ in range(len(A))]
        distances[i][0] = 1.
        visited = [0 for _ in range(len(neigh_list))]
        visit_node(neigh_list, i, visited, distances, P, depth)
        for j in range(len(distances)):
            for k in range(depth + 1):
                distances[j][k] *= pow(alfa, k)
        inverse_p_distances.append([sum(distances[j]) for j in range(len(distances))])
    return np.array(inverse_p_distances)


def merw_inverse_p_distance(A, alfa=0.7):
    ev, v = power_iteration.power_iteration(A)
    D_v = np.diag(v)
    I = np.identity(len(v))
    matr = alfa * A / ev
    P_d = np.dot(np.dot(matr, np.linalg.pinv(D_v)), np.dot(np.linalg.pinv(I - matr), D_v))
    return P_d


def visit_node(neigh_list, u, visited, distances, P, depth):
    visited[u] = 1
    for vertex in neigh_list[u]:
        if visited[vertex] == 0 and depth > 0:
            distances[vertex][5 - depth + 1] = distances[u][5 - depth] * P[u, vertex]
            visit_node(neigh_list, vertex, visited, distances, P, depth - 1)
