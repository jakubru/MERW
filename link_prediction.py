import datetime
import os

import numpy as np

import inverse_p_distance
import sim_rank


class LinkPrediction:

    def __init__(self, graph, approach='MERW', method='laplacians', **kwargs):
        self.approach = approach
        self.method = method
        self.graph = graph
        self.kwargs = kwargs

    def pred(self, edges_percent=0.1):
        edges_idx = np.argwhere(self.graph > 0)
        edges_to_delete = np.random.randint(0, len(edges_idx), int(edges_percent * len(edges_idx)))
        graph_removed_edges = self.graph.copy()
        removed_indices = edges_idx[edges_to_delete]
        graph_removed_edges[removed_indices[:, 0], removed_indices[:, 1]] = 0
        if self.approach == 'MERW':
            preds_idx = self._pred_merw(graph_removed_edges, removed_indices)
        elif self.approach == 'TRW':
            preds_idx = self._pred_trw(graph_removed_edges, removed_indices)
        else:
            raise RuntimeError("Link prediction approach must be set to 'MERW' or 'TRW'")
        preds_idx = np.array(preds_idx)

        graph_removed_edges[preds_idx[:, 0], preds_idx[:, 1]] = 1

        score = self.score(graph_removed_edges, removed_indices)
        return preds_idx, score

    def _pred_merw(self, graph_removed_edges, removed_indices):
        if self.method == 'laplacians':
            raise NotImplementedError
            # return self._pred_laplacians_merw(graph_removed_edges, removed_indices)
        elif self.method == 'simrank':
            return self._pred_simrank_merw(graph_removed_edges, removed_indices)
        elif self.method == 'inv_p_dist':
            return self._pred_inv_p_dist_merw(graph_removed_edges, removed_indices)
        else:
            raise RuntimeError("Link prediction method must be set to 'laplacians', 'simrank' or 'inv_p_dist'")

    def _pred_trw(self, graph_removed_edges, removed_indices):
        if self.method == 'laplacians':
            raise NotImplementedError
            # return self._pred_laplacians_trw(graph_removed_edges, removed_indices)
        elif self.method == 'simrank':
            return self._pred_simrank_trw(graph_removed_edges, removed_indices)
        elif self.method == 'inv_p_dist':
            return self._pred_inv_p_dist_trw(graph_removed_edges, removed_indices)
        else:
            raise RuntimeError("Link prediction method must be set to 'laplacians', 'simrank' or 'inv_p_dist'")

    def _pred_laplacians_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        self.save_to_file('laplacians_trw', preds)
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_laplacians_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        self.save_to_file('laplacians_merw', preds)
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_simrank_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = sim_rank.simrank(graph_removed_edges)
        self.save_to_file('simrank_trw', preds)
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_simrank_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = sim_rank.merw_simrank(graph_removed_edges)
        self.save_to_file('simrank_merw', preds)
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_inv_p_dist_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = inverse_p_distance.inverse_p_distance(graph_removed_edges)
        self.save_to_file('inv_p_dist_trw', preds)
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_inv_p_dist_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = inverse_p_distance.merw_inverse_p_distance(graph_removed_edges)
        self.save_to_file('inv_p_dist_merw', preds)
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def score(self, other_graph, removed_indices):
        expected_edges = other_graph[removed_indices[:, 0], removed_indices[:, 1]]
        return np.count_nonzero(expected_edges) / len(expected_edges)

    def save_to_file(self, filename, results):
        date = datetime.datetime.now()
        date_str = '-'.join([str(date.year), str(date.month), str(date.day)])
        time_str = '-'.join([str(date.hour), str(date.minute), str(date.second)])
        os.mkdir(os.path.join('out', date_str + '_' + time_str))
        np.save(os.path.join('out', date_str + '_' + time_str, filename), results)


def _largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.column_stack(np.unravel_index(indices, ary.shape))
