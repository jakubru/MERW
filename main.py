import numpy as np

import sim_rank

import inverse_p_distance

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

matr = np.array([
        [0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]])

print(inverse_p_distance.inverse_p_distance(matr))

