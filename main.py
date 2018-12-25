import numpy as np
from numpy import linalg as LA

with open('facebook_combined.txt', 'r') as log_fp:
    logs = [log.strip() for log in log_fp.readlines()]

logs_tuple = [tuple(log.split(" ")) for log in logs]

w, h = 4039, 4039
arr = [[0 for x in range(w)] for y in range(h)]

for i in logs_tuple:
    arr[int(i[0])][int(i[1])] = 1
    arr[int(i[1])][int(i[0])] = 1

matr = np.matrix(arr)
a, b = LA.eig(matr)
print(a)
print(b)