import power_iteration
import numpy as np

def inverse_p_distance(A):
    P = power_iteration.compute_transition_matrix(A)
    neigh_list = power_iteration.compute_neighbours(A)
    queue = list()
    B = [[0.0 for _ in range(len(A) )] for _ in range(len(A) + 1)]
    distances = [[0.0 for _ in range(len(A))] for _ in range(len(A))]
    for i in range(1):
        colors = [0 for _ in range(len(A))]
        queue.append(i)
        colors[i] = 1
        B[0][i] = 1.0
        j = 1
        while len(queue) != 0:
            u = queue.pop(0)
            for vertex in neigh_list[u]:
                if vertex != i:
                    k = 0
                    while k < j:
                        print(k+1, u, vertex)
                        B[k + 1][vertex] = B[k][u]*P[u, vertex] + B[k + 1][vertex]
                        print(B)
                        k += 1
                if colors[vertex] == 0:
                    colors[vertex] = 1
                    queue.append(vertex)
            colors[u] = 2
            j+=1
    return np.array(distances)

def merw_inverse_p_distance(A, alfa):
    ev, v = power_iteration.power_iteration(A)
    D_v = np.diag(v)
    I = np.identity(len(v))
    matr = alfa*A/ev
    P_d = np.dot(np.dot(matr, np.linalg.inv(D_v)), np.dot(np.linalg.inv(I - matr), D_v ))
    return P_d