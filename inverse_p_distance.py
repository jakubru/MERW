from power_iteration import power_iteration
import numpy as np

def inverse_p_distance(A):
    P = np.zeros(np.shape(A))
    i = 0
    print(P)
    for element in A:
        sum_ = sum(element)
        for j in range(len(element)):
            if(A[i,j] == 1):
                P[i,j] = 1/float(sum_)
        i+=1
    print(P)

def merw_inverse_p_distance(A, alfa):
    ev, v = power_iteration(A)
    D_v = np.diag(v)
    I = np.identity(len(v))
    matr = alfa*A/ev
    print(I - matr)
    P_d = np.dot(np.dot(matr, np.linalg.inv(D_v)), np.dot(np.linalg.inv(I - matr), D_v ))
    print(P_d)
    return P_d