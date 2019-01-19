import numpy as np
import random
import link_prediction


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

# import os
# l = link_prediction.LinkPrediction(matr)
# preds = np.load(os.path.join('out', '2019-1-19_21-1-11', 'inv_p_dist_merw.npy'))
# preds_idx = l._largest_indices(preds, int(0.1 * len(preds)))
# # print(l._largest_indices(preds, int(0.1 * len(preds))))

l = link_prediction.LinkPrediction(matr, method='inv_p_dist')
edges_percent = 0.1 #random.uniform(0, 0.25)
print(f'Removed {edges_percent * 100 }% edges')
preds, score = l.pred(edges_percent=edges_percent)
print(preds)
print(score)


# print(sim_rank.simrank(matr))
