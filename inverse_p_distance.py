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
    colors = [0 for _ in range(len(A))]
    queue = list()
    for i in range(len(A)):
        queue.append(i)
        colors[i] = 1
        while len(queue) != 0:
            queue.pop(0)

def merw_inverse_p_distance(A, alfa):
    ev, v = power_iteration.power_iteration(A)
    D_v = np.diag(v)
    I = np.identity(len(v))
    matr = alfa*A/ev
    P_d = np.dot(np.dot(matr, np.linalg.inv(D_v)), np.dot(np.linalg.inv(I - matr), D_v ))
    return P_d