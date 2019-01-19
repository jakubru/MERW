import numpy as np

import sim_rank

import power_iteration

import inverse_p_distance

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


# matr = np.array([
#         [0,1,0,0,0,0,0,0,0,0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1,0,0,0,0,0,0,0,0,0]])


neighbours = np.count_nonzero(matr, axis=0)


print(sim_rank.simrank(matr))