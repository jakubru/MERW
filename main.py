import numpy as np

import sim_rank

import inverse_p_distance
import power_iteration

import laplacians

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


with open('facebook_combined.txt', 'r') as log_fp:
    logs = [log.strip() for log in log_fp.readlines()]

logs_tuple = [tuple(log.split(" ")) for log in logs]

w, h = 4039, 4039
arr = np.zeros((w, h))

for i in logs_tuple:
    arr[int(i[0])][int(i[1])] = 1
    arr[int(i[1])][int(i[0])] = 1

matr = np.array(arr)

neighbours = np.count_nonzero(matr, axis=0)



L = laplacians.general_graph_laplacian(matr)
P = power_iteration.compute_transition_matrix(matr)
v = power_iteration.compute_stationary_distribution(P)
print(laplacians.hitting_time(L, v))