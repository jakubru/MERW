import numpy as np

import link_prediction


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


with open('facebook_combined.txt', 'r') as log_fp:
    logs = [log.strip() for log in log_fp.readlines()]

logs_tuple = [tuple(log.split(" ")) for log in logs]
idx = np.array(logs_tuple, dtype=np.int64)
# print(sorted(logs_tuple, key=lambda x:int(x[0])))
w, h = 4039, 4039
matr = np.zeros((w, h), dtype=np.int64)
matr[idx[:, 0], idx[:, 1]] = 1
matr[idx[:, 1], idx[:, 0]] = 1
matr = matr[np.any(matr != 0, axis=0)]
matr_T = matr.T[np.any(matr != 0, axis=0)]
matr = matr_T

print(matr)


# matr = np.array([
#         [0 ,1, 0, 0, 0, 0, 0, 0, 0 ,1],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

# neighbours = np.count_nonzero(matr, axis=0)

# import os
# l = link_prediction.LinkPrediction(matr)
# preds = np.load(os.path.join('out', '2019-1-20_10-58-53', 'inv_p_dist_merw.npy'))
# # preds_idx = l._largest_indices(preds, int(0.1 * len(preds)))
# print(l._largest_indices(preds, int(0.1 * len(preds))))

# l = link_prediction.LinkPrediction(matr, approach='TRW', method='simrank')
# edges_percent = 0.01  # random.uniform(0, 0.25)
# print(f'Removed {edges_percent * 100 }% edges')
# preds, score = l.pred(edges_percent=edges_percent)
# print(preds)
# print(score)

# laplacian_type = 'me' or ' sym_norm_me'
# metrics = 'hitting_time' or 'commute_time'
print('\tinv_p_dist TRW')
l = link_prediction.LinkPrediction(matr, approach='TRW', method='inv_p_dist')
edges_percent = 0.15  # random.uniform(0, 0.25)
print(f'Removed {edges_percent * 100 }% edges')
preds, score = l.pred(edges_percent=edges_percent)
print(preds)
print(score)
print()
print()
print('\tinv_p_dist MERW')
l = link_prediction.LinkPrediction(matr, approach='MERW', method='inv_p_dist')
edges_percent = 0.15  # random.uniform(0, 0.25)
print(f'Removed {edges_percent * 100 }% edges')
preds, score = l.pred(edges_percent=edges_percent)
print(preds)
print(score)


# print(sim_rank.simrank(matr))
