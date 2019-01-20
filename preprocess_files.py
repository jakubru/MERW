import os

import numpy as np
import pandas


def preprocess_file(path):
    df = pandas.read_csv(path, sep='\s+', header=None)
    df.columns = ['from', 'to']
    df.sort_values(by=['from', 'to'], inplace=True)

    df2 = df.copy()
    df2['from'], df2['to'] = df['to'], df['from']
    all_edges = df.append(pandas.DataFrame(df2)).drop_duplicates()

    max_index = df.max() - df.min()
    indexes = list(range(max_index.max()))
    index = 0
    prev = -1
    for vertex in all_edges['from']:
        if vertex != prev:
            all_edges.replace(vertex, indexes[index], inplace=True)
            index += 1
        prev = vertex
    return np.array(all_edges)


if __name__ == '__main__':
    root = 'facebook'
    for path in os.listdir(root):
        if '.edges' in path:
            print(preprocess_file(os.path.join(root, path)))
            print()