import datetime
import os

import numpy as np

import inverse_p_distance
import laplacians
import sim_rank


class LinkPrediction:

    def __init__(self, graph, approach='MERW', method='laplacians', **kwargs):
        self.approach = approach
        self.method = method
        self.graph = graph
        self.kwargs = kwargs

    def pred(self, edges_percent=0.1):
        while True:
            edges_idx = np.argwhere(self.graph > 0)
            edges_to_delete = np.random.randint(0, len(edges_idx), int(edges_percent / 2 * len(edges_idx)))
            graph_removed_edges = self.graph.copy()
            removed_indices = edges_idx[edges_to_delete]
            sum_row = np.sum(self.graph, axis=1)
            actual_edges = []
            for i, j in removed_indices:
                if sum_row[i] > 1 and sum_row[j] > 1:
                    graph_removed_edges[i, j] = 0
                    graph_removed_edges[j, i] = 0
                    actual_edges.append([i, j])
                    sum_row[i] -= 1
                    sum_row[j] -= 1
            actual_edges = np.array(actual_edges)
            print(f'Removed {len(actual_edges)/len(edges_idx)} edges')
            no_edges_idx = np.argwhere(self.graph == 0)
            no_edges = np.random.randint(0, len(no_edges_idx), len(actual_edges))
            no_edges = no_edges_idx[no_edges]
            try:
                if self.approach == 'MERW':
                    preds = self._pred_merw(graph_removed_edges)
                elif self.approach == 'TRW':
                    preds = self._pred_trw(graph_removed_edges)
                else:
                    raise RuntimeError("Link prediction approach must be set to 'MERW' or 'TRW'")
                preds = np.array(preds)
            except Exception as e:
                print(e)
                print('Retrying...')
                continue
            score = self.score(preds, actual_edges, no_edges)
            return preds, score

    def _pred_merw(self, graph_removed_edges):
        if self.method == 'laplacians':
            return self._pred_laplacians_merw(graph_removed_edges)
        elif self.method == 'simrank':
            return self._pred_simrank_merw(graph_removed_edges)
        elif self.method == 'inv_p_dist':
            return self._pred_inv_p_dist_merw(graph_removed_edges)
        else:
            raise RuntimeError("Link prediction method must be set to 'laplacians', 'simrank' or 'inv_p_dist'")

    def _pred_trw(self, graph_removed_edges):
        if self.method == 'laplacians':
            return self._pred_laplacians_trw(graph_removed_edges)
        elif self.method == 'simrank':
            return self._pred_simrank_trw(graph_removed_edges)
        elif self.method == 'inv_p_dist':
            return self._pred_inv_p_dist_trw(graph_removed_edges)
        else:
            raise RuntimeError("Link prediction method must be set to 'laplacians', 'simrank' or 'inv_p_dist'")

    def _pred_laplacians_trw(self, graph_removed_edges):
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
        return preds

    def _pred_laplacians_merw(self, graph_removed_edges):
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
        return preds

    def _pred_simrank_trw(self, graph_removed_edges):
        preds = sim_rank.simrank(graph_removed_edges)
        self.save_to_file('simrank_trw', preds)
        return preds

    def _pred_simrank_merw(self, graph_removed_edges):
        preds = sim_rank.merw_simrank(graph_removed_edges)
        self.save_to_file('simrank_merw', preds)
        return preds

    def _pred_inv_p_dist_trw(self, graph_removed_edges):
        preds = inverse_p_distance.inverse_p_distance(graph_removed_edges)
        self.save_to_file('inv_p_dist_trw', preds)
        return preds

    def _pred_inv_p_dist_merw(self, graph_removed_edges):
        preds = inverse_p_distance.merw_inverse_p_distance(graph_removed_edges)
        self.save_to_file('inv_p_dist_merw', preds)
        return preds

    def score(self, preds, actual_edges, no_edges):
        actual_edges = np.array(actual_edges)
        no_edges = np.array(no_edges)
        print(actual_edges)
        print(no_edges)
        return np.sum(
            preds[actual_edges[:, 0], actual_edges[:, 1]] > preds[no_edges[:, 0], no_edges[:, 1]]
        ) / len(actual_edges)

    def save_to_file(self, filename, results):
        date = datetime.datetime.now()
        date_str = '-'.join([str(date.year), str(date.month), str(date.day)])
        time_str = '-'.join([str(date.hour), str(date.minute), str(date.second)])
        os.mkdir(os.path.join('out', date_str + '_' + time_str))
        np.save(os.path.join('out', date_str + '_' + time_str, filename), results)
