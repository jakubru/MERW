import power_iteration
import numpy as np

def inverse_p_distance(A):
    P = np.zeros(np.shape(A))
    i = 0
    for element in A:
        sum_ = sum(element)
        for j in range(len(element)):
            if(A[i,j] == 1):
                P[i,j] = 1/float(sum_)
        i+=1
    neigh_list = power_iteration.compute_neighbours(A)
    queue = list()
    B = [[0.0 for _ in range(len(A) )] for _ in range(len(A) + 1)]
    distances = [[0.0 for _ in range(len(A))] for _ in range(len(A))]
    for i in range(len(A)):
        colors = [0 for _ in range(len(A))]
        queue.append(i)
        colors[i] = 1
        B[0][i] = 1.0
        j = 1
        while len(queue) != 0:
            u = queue.pop(0)
            for vertex in neigh_list[u]:
                k = 0
                while k < j:
                    for l in range(len(B[j])):
                        B[k + 1][vertex] = B[k][l]*P[vertex,u]
                    k += 1
                if colors[vertex] == 0:
                    colors[vertex] = 1
                    queue.append(vertex)
            colors[u] = 2
            j+=1
        print(B)
    return np.array(distances)

def merw_inverse_p_distance(A, alfa):
    ev, v = power_iteration.power_iteration(A)
    D_v = np.diag(v)
    I = np.identity(len(v))
    matr = alfa*A/ev
    P_d = np.dot(np.dot(matr, np.linalg.inv(D_v)), np.dot(np.linalg.inv(I - matr), D_v ))
    return P_d