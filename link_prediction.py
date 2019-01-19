import numpy as np

import laplacians
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
        graph_removed_edges = a.copy()
        removed_indices = edges_idx[edges_to_delete]
        graph_removed_edges[removed_indices] = 0
        if self.approach == 'MERW':
            preds = self._pred_merw(graph_removed_edges, removed_indices)
        elif self.approach == 'TRW':
            preds = self._pred_trw(graph_removed_edges, removed_indices)
        else:
            raise RuntimeError("Link prediction approach must be set to 'MERW' or 'TRW'")
        score = self.score(preds, removed_indices)
        return preds, score

    def _pred_merw(self, graph_removed_edges, removed_indices):
        if self.method == 'laplacians':
            return self._pred_laplacians_merw(graph_removed_edges, removed_indices)
        elif self.method == 'simrank':
            return self._pred_simrank_merw(graph_removed_edges, removed_indices)
        elif self.method == 'inv_p_dist':
            return self._pred_inv_p_dist_merw(graph_removed_edges, removed_indices)
        else:
            raise RuntimeError("Link prediction method must be set to 'laplacians', 'simrank' or 'inv_p_dist'")

    def _pred_trw(self, graph_removed_edges, removed_indices):
        if self.method == 'laplacians':
            return self._pred_laplacians_trw(graph_removed_edges, removed_indices)
        elif self.method == 'simrank':
            return self._pred_simrank_trw(graph_removed_edges, removed_indices)
        elif self.method == 'inv_p_dist':
            return self._pred_inv_p_dist_trw(graph_removed_edges, removed_indices)
        else:
            raise RuntimeError("Link prediction method must be set to 'laplacians', 'simrank' or 'inv_p_dist'")

    def _pred_laplacians_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_laplacians_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_simrank_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_simrank_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_inv_p_dist_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def _pred_inv_p_dist_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = ...
        new_edges = _largest_indices(preds, n_preds)
        return new_edges

    def score(self, predictions, actual):
        return (predictions == actual).sum() / len(predictions)


def _largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
