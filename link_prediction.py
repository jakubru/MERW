import datetime
import os
import warnings

import numpy as np

import inverse_p_distance
import laplacians
import sim_rank

warnings.filterwarnings('error', message=r'.*?divide by zero.*?')

class LinkPrediction:

    def __init__(self, graph, approach='MERW', method='laplacians', **kwargs):
        self.approach = approach
        self.method = method
        self.graph = graph
        self.kwargs = kwargs

    def pred(self, edges_percent=0.1):
        retry = 0
        while True:
            edges_idx = np.argwhere(self.graph > 0)
            edges_to_delete = np.random.randint(0, len(edges_idx), int(edges_percent / 2 * len(edges_idx)))
            graph_removed_edges = self.graph.copy()
            removed_indices = edges_idx[edges_to_delete]
            graph_removed_edges[removed_indices[:, 0], removed_indices[:, 1]] = 0
            graph_removed_edges[removed_indices[:, 1], removed_indices[:, 0]] = 0
            print('Retry: {}, det original graph == {}'.format(retry, np.linalg.det(self.graph)))
            try:
                if self.approach == 'MERW':
                    preds_idx = self._pred_merw(graph_removed_edges, removed_indices)
                elif self.approach == 'TRW':
                    preds_idx = self._pred_trw(graph_removed_edges, removed_indices)
                else:
                    raise RuntimeError("Link prediction approach must be set to 'MERW' or 'TRW'")
                preds_idx = np.array(preds_idx)
                graph_removed_edges[preds_idx[:, 0], preds_idx[:, 1]] = 1
                graph_removed_edges[preds_idx[:, 1], preds_idx[:, 0]] = 1
                score = self.score(graph_removed_edges, removed_indices)
                return preds_idx, score
            except RuntimeWarning as w:
                retry += 1
                print(w)

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
        metrics = self.kwargs.get('metrics', None)
        L = laplacians.general_graph_laplacian(graph_removed_edges)
        if metrics == 'hitting_time':
            preds = laplacians.hitting_time(L)
        elif metrics == 'commute_time':
            preds = laplacians.commute_time(L)
        elif metrics == 'commute_kernel':
            preds = laplacians.commute_kernel(L)
        else:
            raise RuntimeError('Laplacian metrics must be "hitting_time", "commute_kernel" or "commute_time"')
        self.save_to_file('laplacians_trw', preds)
        new_edges = self._largest_indices(preds, n_preds)
        return new_edges

    def _pred_laplacians_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        metrics = self.kwargs.get('metrics', None)
        laplacian_type = self.kwargs.get('laplacian_type', None)
        if laplacian_type == 'me':
            L = laplacians.me_combinatorial_graph_laplacian(graph_removed_edges)
        elif laplacian_type == 'sym_norm_me':
            L = laplacians.sym_norm_me_graph_laplacian(graph_removed_edges)
        else:
            raise RuntimeError('Laplacian type must be "me" or "sym_norm_me"')
        if metrics == 'hitting_time':
            preds = laplacians.hitting_time(L)
        elif metrics == 'commute_time':
            preds = laplacians.commute_time(L)
        elif metrics == 'commute_kernel':
            preds = laplacians.commute_kernel(L)
        else:
            raise RuntimeError('Laplacian metrics must be "hitting_time", "commute_kernel" or "commute_time"')
        self.save_to_file('laplacians_merw', preds)
        new_edges = self._largest_indices(preds, n_preds)
        return new_edges

    def _pred_simrank_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = sim_rank.simrank(graph_removed_edges)
        for i in range (len(preds)):
            preds[i,i] = 0.0
        self.save_to_file('simrank_trw', preds)
        new_edges = self._largest_indices(preds, n_preds)
        return new_edges

    def _pred_simrank_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = sim_rank.merw_simrank(graph_removed_edges)
        for i in range (len(preds)):
            preds[i,i] = 0.0
        self.save_to_file('simrank_merw', preds)
        new_edges = self._largest_indices(preds, n_preds)
        return new_edges

    def _pred_inv_p_dist_trw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = inverse_p_distance.inverse_p_distance(graph_removed_edges)
        for i in range (len(preds)):
            preds[i,i] = 0.0
        self.save_to_file('inv_p_dist_trw', preds)
        new_edges = self._largest_indices(preds, n_preds)
        return new_edges

    def _pred_inv_p_dist_merw(self, graph_removed_edges, removed_indices):
        n_preds = len(removed_indices)
        preds = inverse_p_distance.merw_inverse_p_distance(graph_removed_edges)
        for i in range(len(preds)):
            preds[i,i] = 0.0
        self.save_to_file('inv_p_dist_merw', preds)
        new_edges = self._largest_indices(preds, n_preds)
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

    def _largest_indices(self, preds, n):
        """Returns the n largest indices from a numpy array."""
        no_edges = np.argwhere(self.graph == 0)
        no_edges_preds = preds[no_edges[:, 0], no_edges[:, 1]]
        no_edges_preds = no_edges_preds[np.argwhere(no_edges[:, 0] != no_edges[:, 1])]
        flat = no_edges_preds.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.column_stack(np.unravel_index(indices, preds.shape))
