import numpy as np
import math
import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib.gridspec import GridSpec

START_PLOT = 0
CONTINUATION_DEBUG = False
PROXIMITY_SHOW = False
PROXIMITY_DEBUG = False
SAVE_FIG = False
LOCALITY = True
WITHLINE = False
if SAVE_FIG:
    matplotlib.use("Agg")


class Perception:

    def __init__(self, log_path=None, k=None):
        self.history_info = {
            'inter_intra_dist_ratio': [],
            'point_base_dist_Ratio': [],
            'statics': []
        }
        self.continuation_data = {
            'red std': [],
            'green std': [],
            'red N': [],
            'green N': [],
            'mean std': [],
            'std std': [],
            'std ratio': [],
            'threshold': [],
            'continuation': [],
            'merged': [],
        }
        if log_path is not None:
            self.log_path = log_path
        else:
            self.log_path = 'plots'

        self.THRESHOLD_CONTINUATION = 0.15
        self.NUM_NEAREST_CLUSTERS = 5
        self.k = k

    def fit(self, X):
        self.X = X
        self.DIMENSION = len(X[0])
        self.X_RANGE = min(X[:, 0]), max(X[:, 0])
        self.Y_RANGE = min(X[:, 1]), max(X[:, 1])
        self.DIST_MATRIX = np.full((len(X), len(X)), np.inf)
        self.FAIL_MATRIX = np.empty((len(X), len(X), 2))

        self.initial_clusters = {
            id: {
                'label': id,
                'data': [id],
                'center': x,
                'mean_dist': 0,
                'std_dist': 0,
                'past_dists': [],
                'merging_dists': [],
                'past_densities': [],
                'past_std': [],
                'traces': []
            }
            for id, x in enumerate(self.X)}

        start_time = time.time()
        self.clusters_dist, self.points_dist = self.initiate_dists()
        early_stop = False

        while len(self.clusters_dist):
            i = 0
            # sorted_clusters_dist = sorted(self.clusters_dist.items(), key=lambda item: item[1])
            pre_length = len(self.initial_clusters)

            k_nearest_clusters = self.get_indices_of_k_smallest(k=X.shape[0], matrix=self.DIST_MATRIX, sorted=True)
            # self.GRAPH = self.construct_graph(k_nearest_clusters)
            # k_nearest_clusters = self.get_indices_of_k_smallest(k=(pre_length * pre_length) // 4, matrix=self.DIST_MATRIX, sorted=True)
            last_merged_cluster = None

            while i < len(k_nearest_clusters[0]):
                c1, c2 = k_nearest_clusters[:, i]
                print(f"Round: {i} -- {c1} and {c2}")
                if self.k is not None:
                    early_stop = self.k == len(self.initial_clusters)
                    if early_stop:
                        break

                # if self.DIST_MATRIX[c1][c2] > 100 * self.MIN_BTN_CLUSTER_DIST:
                #     break

                # if c1 in self.initial_clusters.keys() and c2 in self.initial_clusters.keys():
                if ~np.isinf(self.DIST_MATRIX[c1][c2]):
                    # to avoid replicated merging of two failed clusters.
                    # n_clusters = len(self.initial_clusters[c1]['data']) + len(self.initial_clusters[c2]['data'])
                    # if self.FAIL_MATRIX[c1][c2][0] == n_clusters and self.FAIL_MATRIX[c1][c2][1]:
                    #     i += 1
                    #     continue

                    if last_merged_cluster is not None:
                        _c = self.get_indices_of_k_smallest(1, self.DIST_MATRIX[last_merged_cluster, :])[:, 0][0]

                        if self.DIST_MATRIX[last_merged_cluster][_c] < self.DIST_MATRIX[c1][c2]:
                            c1, c2 = last_merged_cluster, _c
                        else:
                            i += 1
                        last_merged_cluster = None
                    else:
                        i += 1

                    is_complete, merged_cluster = self.merge_clusters(c1, c2)
                    if is_complete:
                        last_merged_cluster = merged_cluster
                        if len(self.initial_clusters) < START_PLOT:
                            self.plot(WITHLINE)
                            if self.DIMENSION > 3:
                                self.save_clusters()
                else:
                    i += 1

            print(f"The number of existing clusters: {len(self.initial_clusters)}")
            if pre_length == len(self.initial_clusters) or early_stop:
                break
        if self.k is not None:
            if self.k < len(self.initial_clusters):
                Y_hat = np.empty(len(self.X))
                for k, v in self.initial_clusters.items():
                    for x in v['data']:
                        Y_hat[x] = int(v['label'])

                res = np.column_stack((self.X, Y_hat))
                np.savetxt(f'/k.out', res)

                self.plot(withline=True)
                self.post_processing()

        print(f"Clustering finished in {(time.time() - start_time) / 60} minutes")
        Y_hat = np.empty(len(self.X))
        for k, v in self.initial_clusters.items():
            for x in v['data']:
                Y_hat[x] = int(v['label'])
        return Y_hat, self.X

    def construct_graph(self, cluster_pairs):
        existing_clusters = []
        pairs = []
        for i in range(len(cluster_pairs[0])):
            if cluster_pairs[0][i] not in existing_clusters or cluster_pairs[1][i] not in existing_clusters:
                pairs.append([cluster_pairs[0][i], cluster_pairs[1][i],
                              self.DIST_MATRIX[cluster_pairs[0][i]][cluster_pairs[1][i]]])
                if cluster_pairs[0][i] not in existing_clusters:
                    existing_clusters.append(cluster_pairs[0][i])
                if cluster_pairs[1][i] not in existing_clusters:
                    existing_clusters.append(cluster_pairs[1][i])
        return np.array(pairs)

    def post_processing(self):
        if self.k is not None:
            ranking_clusters = list(sorted(self.initial_clusters.items(), key=lambda x: len(x[1]['data'])))
            top_k_clusters = ranking_clusters[-self.k:]
            remaining_clusters = ranking_clusters[:len(ranking_clusters) - self.k]
            merging_pairs = []
            for c in remaining_clusters:
                cluster_dists = [(t[0], self.DIST_MATRIX[t[0]][c[0]]) for t in top_k_clusters]
                nearest_c = sorted(cluster_dists, key=lambda x: x[1])[0]
                merging_pairs.append([nearest_c[0], c[0]])
            for p in merging_pairs:
                self.merge_2_cluster(p[0], p[1])

            print(f'Post processing completed: {self.k} clusters achieved!')

    def initiate_dists(self):
        clusters_dists = {}
        points_dists = {}

        min_dist = np.inf
        sum_dist = 0
        count = 0
        DISTS = []

        list_clusters = list(self.initial_clusters.keys())
        for i in range(len(list_clusters)):
            for j in range(i + 1, len(list_clusters)):
                _d = self.compute_2_clusters_dist(list_clusters[i], list_clusters[j])
                clusters_dists[frozenset([list_clusters[i],
                                 list_clusters[j]])] = {
                    'distance_info': _d
                }
                _d = _d['near_dist']['distance']
                self.DIST_MATRIX[j][i], self.DIST_MATRIX[i][j] = _d, _d

                if list_clusters[i] in points_dists.keys():
                    points_dists[list_clusters[i]].append([list_clusters[j], _d])
                else:
                    points_dists[list_clusters[i]] = [[list_clusters[j], _d]]
                if list_clusters[j] in points_dists.keys():
                    points_dists[list_clusters[j]].append([list_clusters[i], _d])
                else:
                    points_dists[list_clusters[j]] = [[list_clusters[i], _d]]

                DISTS.append(_d)

        DISTS = sorted(DISTS)
        # to avoid extremly small min inter-clusters distance
        # for i in range(1, len(DISTS)):
        #     if DISTS[i] / DISTS[i-1] < 1.5:
        #         self.MIN_BTN_CLUSTER_DIST = DISTS[i]
        #         break
        self.MIN_BTN_CLUSTER_DIST = DISTS[int(len(DISTS)/100)]
        # self.MEAN_BTN_CLUSTER_DIST = np.mean(DISTS)
        self.MEAN_WTN_CLUSTER_DIST = 0.0001

        for k, v in points_dists.items():
            points_dists[k] = list(sorted(v, key=lambda x: x[1]))

        return clusters_dists, points_dists

    def compute_2_clusters_dist(self, c1, c2):
        '''
        This is to calculate the distance between two clusters, c1 and c2.
        The easiest way is the distance between two centers.
        :param c1:
        :param c2:
        :return: distance
        '''

        closest_points = self.closest_points_btn_2_clusters(c1, c2)
        near_dist = np.linalg.norm(self.X[closest_points[0]] - self.X[closest_points[1]]), closest_points[0], closest_points[1]

        center_dist = np.linalg.norm(self.initial_clusters[c1]['center'] - self.initial_clusters[c2]['center']), \
               self.initial_clusters[c1]['center'], self.initial_clusters[c2]['center']

        mix_dist = (near_dist[0] + center_dist[0]) / 2, None, None

        dist = {
            'near_dist': {'distance': near_dist[0], 'reference_points': {c1: near_dist[1], c2: near_dist[2]}},
            'center_dist': {'distance': center_dist[0], 'reference_points': {c1: center_dist[1], c2: center_dist[2]}},
            'mix_dist': {'distance': mix_dist[0], 'reference_points': {c1: mix_dist[1], c2: mix_dist[2]}}
        }

        return dist

    def get_indices_of_k_smallest(self, k, matrix, sorted=False):
        idx = np.argpartition(matrix.ravel(), k)
        ind = np.array(np.unravel_index(idx, matrix.shape))[:, range(min(k, 0), max(k, 0))]
        if sorted:
            values = matrix[tuple(ind)]
            xx = np.argsort(values)
            return ind[:, xx]
        else:
            return ind
        # if you want it in a list of indices . . .
        # return np.array(np.unravel_index(idx, arr.shape))[:, range(k)].transpose().tolist()

    def merge_clusters(self, cluster1, cluster2):
        '''
        Merge two clusters. There are three cases: 1) point-point; 2) point-cluster; 3) cluster-cluster.

        :param cluster1:
        :param cluster2:
        :return:
        '''
        print('\nExisting clusters: ', len(self.initial_clusters), f'Merging clusters: {cluster1} and {cluster2}')
        is_merge, lead_cluster, child_cluster = self.vision_generic(cluster1, cluster2)
        if is_merge:
            self.history_info['inter_intra_dist_ratio'].append(1)
            print(f"Succeed. Current clusters info: {lead_cluster} and {child_cluster} merged!")
            self.merge_2_cluster(lead_cluster, child_cluster)
            return True, lead_cluster
        print('Failed.')
        return False, None

    def update_clusters(self, new_c, old_c):
        del self.initial_clusters[old_c]
        self.DIST_MATRIX[:, old_c] = np.inf
        self.DIST_MATRIX[old_c, :] = np.inf

        # update clusters distance
        existing_clusters = np.argwhere(~np.isinf(self.DIST_MATRIX[new_c, :])).reshape((-1))
        for c in existing_clusters:
            key = frozenset((new_c, c))
            _d = self.compute_2_clusters_dist(new_c, c)
            self.clusters_dist[key]['distance_info'] = _d
            self.DIST_MATRIX[new_c][c], self.DIST_MATRIX[c][new_c] = _d['near_dist']['distance'], \
                                                             _d['near_dist']['distance']
        self.MIN_BTN_CLUSTER_DIST = max(self.DIST_MATRIX.min(), self.MIN_BTN_CLUSTER_DIST)
        # self.MIN_BTN_CLUSTER_DIST = self.DIST_MATRIX.min()

    def merge_2_cluster(self, cluster1_id, cluster2_id):
        cluster1, cluster2 = self.initial_clusters[cluster1_id], self.initial_clusters[cluster2_id]
        p1, p2 = self.clusters_dist[frozenset((cluster1_id, cluster2_id))]['distance_info']['near_dist']['reference_points'][cluster1_id], \
                        self.clusters_dist[frozenset((cluster1_id, cluster2_id))]['distance_info']['near_dist']['reference_points'][
                            cluster2_id]
        cluster1['traces'].append((p1, p2))
        cluster1['traces'].extend(cluster2['traces'])
        cluster1['data'].extend(cluster2['data'])
        cluster1['center'] = np.mean(self.X[cluster1['data']], axis=0)
        dist_data = np.linalg.norm(self.X[cluster1['data']] - cluster1['center'], axis=1)
        cluster1['mean_dist'] = np.mean(dist_data)
        cluster1['std_dist'] = np.std(dist_data)
        cluster1['past_std'].append(cluster1['std_dist'])
        cluster1['past_dists'].extend(cluster2['past_dists'])
        self.update_clusters(cluster1_id, cluster2_id)

    def vision_generic(self, cluster1, cluster2):
        proximity, distance, lead_cluster, child_cluster = self.compute_proximity(cluster1, cluster2)

        adaptive_proximity_T1, adaptive_proximity_T2, shape_diff, adp_prox = self.compute_adaptive_threshold(self.NUM_NEAREST_CLUSTERS,
                                                               lead_cluster, child_cluster, distance, proximity)
        # adaptive_proximity_T1, adaptive_proximity_T2, shape_diff = 1, 1, 1
        print(f'--Adaptive threshold: {proximity}, {adaptive_proximity_T1}, {adaptive_proximity_T2}')

        # if proximity > adaptive_proximity_T:  # self.THRESHOLD_ROXIMITY
        if proximity > adaptive_proximity_T1 or proximity > adaptive_proximity_T2:
            # self.continuation_data['continuation'].append(0)
            self.continuation_data['merged'].append(False)
            return False, cluster1, cluster2

        continuation = self.compute_continuation(lead_cluster, child_cluster,
                                                 self.THRESHOLD_CONTINUATION / shape_diff,
                                                 distance / proximity, adp_prox)# max((adaptive_proximity_T1, adaptive_proximity_T2)))#(adaptive_proximity_T1 + adaptive_proximity_T2) / 2)
        self.continuation_data['continuation'].append(round(continuation, 3))

        if continuation > self.THRESHOLD_CONTINUATION / shape_diff: # self.THRESHOLD_CONTINUATION
            print(f'--proximity: ', proximity)
            print(f'--Continuation: ', continuation)

            self.initial_clusters[lead_cluster]['past_densities'].append(continuation)
            self.continuation_data['merged'].append(True)

            self.initial_clusters[lead_cluster]['past_dists'].append(distance)
            self.initial_clusters[lead_cluster]['merging_dists'].append(distance)

            return True, lead_cluster, child_cluster

        self.continuation_data['merged'].append(False)

        return False, lead_cluster, child_cluster

    def compute_force_btn_clusters(self, cluster1, cluster2):
        dist = self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['mix_dist']['distance']
        m1, m2 = len(self.initial_clusters[cluster1]['data']), len(self.initial_clusters[cluster2]['data'])
        f = m1 * m2 / (dist**2)
        return f

    def compute_adaptive_threshold(self, k, cluster1, cluster2, distance, proximity):
        size1, size2 = len(self.initial_clusters[cluster1]['data']), len(self.initial_clusters[cluster2]['data'])
        # Step 1 select contextual clusters
        k = min(len(self.initial_clusters) - 1, k)
        nearest_clusters1 = self.get_indices_of_k_smallest(k, self.DIST_MATRIX[cluster1, :], sorted=True)
        nearest_clusters2 = self.get_indices_of_k_smallest(k, self.DIST_MATRIX[cluster2, :], sorted=True)

        nearest_dists1 = self.DIST_MATRIX[cluster1, nearest_clusters1][0]
        nearest_dists2 = self.DIST_MATRIX[cluster2, nearest_clusters2][0]
        nearest_dists1 = np.delete(nearest_dists1, np.where(nearest_dists1 == np.Inf))
        nearest_dists2 = np.delete(nearest_dists2, np.where(nearest_dists2 == np.Inf))

        mean_e, std_e = np.mean([*nearest_dists1, *nearest_dists2]), np.std([*nearest_dists1, *nearest_dists2])

        base_force = self.compute_force_btn_clusters(cluster1, cluster2)
        forces = [(1, cluster2)]
        for c in nearest_clusters1[0]:
            if c in nearest_clusters2[0]:
                _force1 = self.compute_force_btn_clusters(c, cluster1)
                _force2 = self.compute_force_btn_clusters(c, cluster2)
                forces.append((math.sqrt(_force1 * _force2) / base_force, c))

        if len(forces):
            forces1 = sorted(forces, key=lambda x: x[0], reverse=True)
            cont_clusters = []
            N_rc = 2
            Total_affect = 0
            for rc in forces1:
                # if len(self.initial_clusters[rc[1]]['data']) >= max(size1, size2):
                #     cont_clusters.append(rc[1])
                #     if len(cont_clusters) == 2:
                #         break
                dist_c_cluster1 = \
                    self.clusters_dist[frozenset((rc[1], cluster1))]['distance_info']['near_dist']['distance']
                dist_c_cluster2 = \
                    self.clusters_dist[frozenset((rc[1], cluster2))]['distance_info']['near_dist']['distance'] \
                        if rc[1] != cluster2 else dist_c_cluster1
                cont_clusters.append((rc[0], (dist_c_cluster2 + dist_c_cluster1) / 2))
                Total_affect += rc[0]
                if len(cont_clusters) == N_rc:
                    break

            vision_scale = 0
            for rc in cont_clusters:
                dist_ratio = distance / rc[1] if rc[1] != 0 else 1
                _vision_scale = 2 / (1 + math.e ** (5 * dist_ratio)) + 1
                vision_scale += _vision_scale * rc[0] / Total_affect

        else:
            vision_scale = 1
            dist_ratio, dist_c_cluster1, dist_c_cluster2 = 1, 1, 1

        if len(self.initial_clusters[cluster1]['past_dists']) > 3:
            mean1, std1 = np.mean(self.initial_clusters[cluster1]['past_dists']), \
                          np.std(self.initial_clusters[cluster1]['past_dists'])
            threshold1 = 3 / (1 + math.e ** (0.3 * mean1 / std1)) + 1.4 if std1 != 0 and mean1 != 0 else 2.7
        else:
            threshold1 = 2.7
        threshold1 = vision_scale*threshold1

        if len(self.initial_clusters[cluster2]['past_dists']) > 3:
            mean2, std2 = np.mean(self.initial_clusters[cluster2]['past_dists']), \
                          np.std(self.initial_clusters[cluster2]['past_dists'])
            # threshold2 = (1 + mean2 / std2) ** (std2 / mean2) if std2 != 0 and mean2 != 0 else 2.7
            threshold2 = 3 / (1 + math.e ** (0.3 * mean2 / std2)) + 1.4 if std2 != 0 and mean2 != 0 else 2.7
        else:
            threshold2 = 2.7
        threshold2 = vision_scale*threshold2

        N1, N2 = len(self.initial_clusters[cluster1]['past_std']), \
                 len(self.initial_clusters[cluster2]['past_std'])
        N = min(N1, N2)
        if len(self.initial_clusters[cluster2]['past_std']) > 1:
            m5 = np.mean(self.initial_clusters[cluster1]['past_std'][-N:])
            m6 = np.mean(self.initial_clusters[cluster2]['past_std'][-N:])

            Nt = N
            if N1 < N2:
                for i in range(N-1, N2):
                    if self.initial_clusters[cluster2]['past_std'][i] >= self.initial_clusters[cluster1]['past_std'][-1]:
                        Nt = i + 1
                        break

                m1 = np.mean(self.initial_clusters[cluster1]['past_std'][:N])
                m2 = np.mean(self.initial_clusters[cluster2]['past_std'][:Nt])
                std1 = np.std(self.initial_clusters[cluster1]['past_std'][:N]) * m1
                std2 = np.std(self.initial_clusters[cluster2]['past_std'][:Nt]) * m2

            else:
                for i in range(N-1, N1):
                    if self.initial_clusters[cluster1]['past_std'][i] >= self.initial_clusters[cluster2]['past_std'][-1]:
                        Nt = i + 1
                        break

                m1 = np.mean(self.initial_clusters[cluster1]['past_std'][:Nt])
                m2 = np.mean(self.initial_clusters[cluster2]['past_std'][:N])
                std1 = np.std(self.initial_clusters[cluster1]['past_std'][:Nt]) * m1
                std2 = np.std(self.initial_clusters[cluster2]['past_std'][:N]) * m2

            most_same_std = round(min(std1, std2) / max(std1, std2), 3) if std1 != 0 and std2 != 0 else 1

            # std3 = np.std(self.initial_clusters[cluster1]['past_std'][:N]) * m3
            # std4 = np.std(self.initial_clusters[cluster2]['past_std'][:N]) * m4
            std5 = np.std(self.initial_clusters[cluster1]['past_std'][-N:]) * m5
            std6 = np.std(self.initial_clusters[cluster2]['past_std'][-N:]) * m6

            l_same_std = round(min(std5, std6) / max(std5, std6), 3) if std5 != 0 and std6 != 0 else 1

        else:
            self.continuation_data['mean std'].append(0)
            self.continuation_data['std std'].append(0)
            self.continuation_data['std ratio'].append(1)
            std1, std2, std3, std4, std5, std6 = 0, 0, 0, 0, 0, 0
            same_std, l_same_std, most_same_std = 1, 1, 1

        _diff = max([l_same_std, most_same_std])
        shape_diff = _diff / (1 + _diff) + 0.5 if N > 4 else 1
        threshold1 *= shape_diff
        threshold2 *= shape_diff

        if PROXIMITY_SHOW and len(self.initial_clusters) < START_PLOT and \
                (proximity <= min(threshold1, threshold2) or PROXIMITY_DEBUG):  #  and cluster and refe

            p1, p2 = self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['reference_points'][cluster1], \
                        self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['reference_points'][
                            cluster2]
            points_dist = np.linalg.norm(self.X[p1] - self.X[p2])
            base_length = points_dist
            radius = (threshold1 + threshold2) * base_length / 2

            if self.DIMENSION == 2:
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

            else:
                f = plt.figure(figsize=(15, 8))
                ax1 = f.add_subplot(1, 3, 1, projection='3d')
                ax2 = f.add_subplot(1, 3, 2)
                ax3 = f.add_subplot(1, 3, 3)

            self.plot_point(self.X, 'k.', axes=ax1)
            self.plot_point(p1, 'r*', axes=ax1)
            self.plot_point(p2, 'g*', axes=ax1)

            red = f"Red cluster: mean = {self.initial_clusters[cluster1]['mean_dist']:.4f}, " \
                  f"std = {self.initial_clusters[cluster1]['std_dist']:.4f}, " \
                  f"\nstd/N = {self.initial_clusters[cluster1]['std_dist'] / len(self.initial_clusters[cluster1]['data']):.4f} , " \
                  f"mean/N = {self.initial_clusters[cluster1]['mean_dist'] / len(self.initial_clusters[cluster1]['data']):.4f}, " \
                  f"N = {len(self.initial_clusters[cluster1]['data'])}"

            green = f"Green cluster: mean = {self.initial_clusters[cluster2]['mean_dist']:.4f}, " \
                  f"std = {self.initial_clusters[cluster2]['std_dist']:.4f}, " \
                  f"\nstd/N = {self.initial_clusters[cluster2]['std_dist'] / len(self.initial_clusters[cluster2]['data']):.4f} , " \
                  f"mean/N = {self.initial_clusters[cluster2]['mean_dist'] / len(self.initial_clusters[cluster2]['data']):.4f}, " \
                  f"N = {len(self.initial_clusters[cluster2]['data'])}"

            self.plot_point(self.initial_clusters[cluster1]['data'], 'r.', axes=ax1)
            self.plot_point(self.initial_clusters[cluster2]['data'], 'g.', axes=ax1)
            if self.DIMENSION == 2:
                ax1.set_aspect('equal', adjustable='box')
                cir = plt.Circle(self.X[p1], radius, color='r', fill=False)
                cir2 = plt.Circle(self.X[p2], radius, color='g', fill=False)
                ax1.add_patch(cir)
                ax1.add_patch(cir2)

                ax1.text(self.X_RANGE[0], self.Y_RANGE[0]-5, red)
                ax1.text(self.X_RANGE[0], self.Y_RANGE[0]-11, green)
                ax1.set_ylim((self.Y_RANGE[0]-12, self.Y_RANGE[1] + 2))
                ax1.set_xlim((self.X_RANGE[0] - 2, max(20, self.X_RANGE[1] + 2)))

            # plot nearest clusters distance
            nearest_clusters1 = self.get_indices_of_k_smallest(k, self.DIST_MATRIX[cluster1, :], sorted=True)
            nearest_clusters2 = self.get_indices_of_k_smallest(k, self.DIST_MATRIX[cluster2, :], sorted=True)

            nearest_dists1 = self.DIST_MATRIX[cluster1, nearest_clusters1][0]
            nearest_dists2 = self.DIST_MATRIX[cluster2, nearest_clusters2][0]

            ax2.plot(range(len(nearest_clusters1[0])),
                     nearest_dists1.tolist(), 'r.-', label='Near cluster of Red')
            ax2.plot(range(len(nearest_clusters2[0])),
                     nearest_dists2.tolist(), 'g.-', label='Near cluster of Green')

            ax2.plot(range(len(self.initial_clusters[cluster1]['past_dists'])),
                     self.initial_clusters[cluster1]['past_dists'], 'r*--', label='Past dists of Red')
            ax2.plot(range(len(self.initial_clusters[cluster2]['past_dists'])),
                     self.initial_clusters[cluster2]['past_dists'], 'g*--', label='Past dists of Green')

            l, h = min(*nearest_dists1[:10], *nearest_dists2[:10]), \
                   max(*nearest_dists1[:10], *nearest_dists2[:10])

            ax2.text(1, l + 0.5 * (h-l),
                     f"Red: Std = {np.std(nearest_dists1[:self.NUM_NEAREST_CLUSTERS]):.2f}.")
            ax2.text(1, l + 0.4 * (h-l),
                     f"Green: Std = {np.std(nearest_dists2[:self.NUM_NEAREST_CLUSTERS]):.2f}.")
            ax2.text(1, l,
                     f"Proximity = {proximity:.3f}; VS: {vision_scale:.3f} \nThreshold = {threshold1:.3f}(R)|{threshold2:.3f}(G)\n "
                     f"Distance: {distance:.3f} : {dist_c_cluster1:.3f}, {dist_c_cluster2:.3f},\n"
                     f"Std: {std_e:.3f}, DR: {dist_ratio:.3f}, / {mean_e / std_e:.3f}"
                    )
            # f"Dist ratio: {dist_ratio:.3f}; \n"
            # f"{distance:.3f} / {dist_c_cluster1:.3f} + {dist_c_cluster2:.3f}; force: {force:.3f}"

            ax2.legend()
            ax2.set_title(f"{k} nearest cluster distances for cluster red and green")

            ax3.plot(range(len(self.initial_clusters[cluster1]['past_std'])),
                     self.initial_clusters[cluster1]['past_std'], 'r*--', label='Past stds of Red')
            ax3.plot(range(len(self.initial_clusters[cluster2]['past_std'])),
                     self.initial_clusters[cluster2]['past_std'], 'g*--', label='Past stds of Green')
            _std1 = np.std(self.initial_clusters[cluster1]['past_dists'])
            _std2 = np.std(self.initial_clusters[cluster2]['past_dists'])
            ax3.text(1, 0, f"Past dists: {_std1:.3f} / {_std2:.3f} = {min(_std1, _std2) / max(_std1, _std2):.3f}\n"
                           f"Past std: {std1:.3f} / {std2:.3f} = {self.continuation_data['std ratio'][-1]:.3f}\n"
                           f"Same std: {1:.3f}; L same: {l_same_std:.3f} M same: {most_same_std:.3f}\n"
                           f"Diff: {_diff:.3f}\n"
                           f"Shape diff: {shape_diff:.3f}\n")
            ax3.legend()

            if SAVE_FIG:
                f.savefig(f'{self.log_path}/{len(self.initial_clusters)}_clusters.png')
            else:
                plt.show()
            plt.close('all')
        return threshold1, threshold2, shape_diff, threshold1 * size1 / (size1 + size2) + \
               threshold2 * size2 / (size1 + size2)

    def compute_proximity(self, cluster1, cluster2):
        distance = self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['distance']
        n1, n2 = len(self.initial_clusters[cluster1]['past_dists']), len(self.initial_clusters[cluster2]['past_dists'])
        n = max(n1//2, n2) if n1 > n2 else max(n2//2, n1)

        mean_past_dist = np.mean([*self.initial_clusters[cluster1]['past_dists'][-n:],
                                  *self.initial_clusters[cluster2]['past_dists'][-n:]]) \
            if n1 > 2 or n2 > 2 else self.MIN_BTN_CLUSTER_DIST

        if len(self.initial_clusters[cluster1]['data']) >= len(self.initial_clusters[cluster2]['data']):
            return distance / mean_past_dist, distance, cluster1, cluster2
        else:
            return distance / mean_past_dist, distance, cluster2, cluster1

    def weighted_mean_dist(self, cluster, reference_p):
        dists = []
        weight_sum = 0
        for p1, p2 in self.initial_clusters[cluster]['traces']:
            weight = max(np.linalg.norm(p1 - reference_p), np.linalg.norm(p2 - reference_p))
            # weight = 1
            dists.append((np.linalg.norm(p1 - p2), weight))
            weight_sum += 1 / weight

        weight_dist = 0
        for d in dists:
            weight_dist += d[0] * (1 / d[1]) / weight_sum
        return weight_dist

    def cluster_2_tree(self, cluster, root):
        root_id = int(np.where((self.initial_clusters[cluster]['data'] == root).all(axis=1))[0][0])

        adj_list = {}
        for p1, p2 in self.initial_clusters[cluster]['traces']:
            p1_id = int(np.where((self.initial_clusters[cluster]['data'] == p1).all(axis=1))[0][0])
            p2_id = int(np.where((self.initial_clusters[cluster]['data'] == p2).all(axis=1))[0][0])
            if p1_id in adj_list.keys():
                adj_list[p1_id].append(p2_id)
            else:
                adj_list[p1_id] = [p2_id]

            if p2_id in adj_list.keys():
                adj_list[p2_id].append(p1_id)
            else:
                adj_list[p2_id] = [p1_id]

        tree = {
            'data': root,
            'id': root_id,
            'children': []
        }

        visited = []
        self.DFS(tree, adj_list, cluster, visited)
        return tree

    def DFS(self, parent, adj_list, cluster, visited):
        if parent['id'] in visited:
            return
        visited.append(parent['id'])
        for child in adj_list[parent['id']]:
            child_node = {
                'data': self.initial_clusters[cluster]['data'][child],
                'id': child,
                'children': []
            }
            parent['children'].append(child_node)
            self.DFS(child_node, adj_list, cluster, visited)

    def find_closet_cluster(self, cluster):
        near_clusters, min_dist = (None, None), np.inf

        for k, v in self.clusters_dist.items():
            if cluster in k:
                if v['distance'] < min_dist:
                    near_clusters, min_dist = k, v
        return near_clusters, min_dist

    def plot_point(self, p, style='b.', axes=plt):

        if isinstance(p, int):
            # if axes:
            if self.DIMENSION == 2:
                axes.plot(self.X[p][0], self.X[p][1], style)
            else:
                axes.plot(self.X[p][0], self.X[p][1], self.X[p][2], style)
            # else:
            #     if self.DIMENSION == 2:
            #         plt.plot(self.X[p][0], self.X[p][1], style)
            #     else:
            #         plt.plot(self.X[p][0], self.X[p][1], self.X[p][2], style)
        elif isinstance(p, np.ndarray):
            if len(p.shape) > 1:
                if self.DIMENSION == 2:
                    x = [_p[0] for _p in p]
                    y = [_p[1] for _p in p]
                    axes.plot(x, y, style)
                else:
                    x = [_p[0] for _p in p]
                    y = [_p[1] for _p in p]
                    z = [_p[2] for _p in p]
                    axes.plot(x, y, z, style)
            else:
                if self.DIMENSION == 2:
                    axes.plot(p[0], p[1], style)
                else:
                    axes.plot(p[0], p[1], p[2], style)

        elif isinstance(p, list):
            if self.DIMENSION == 2:
                x = [self.X[int(_p)][0] for _p in p]
                y = [self.X[int(_p)][1] for _p in p]
                axes.plot(x, y, style)
            else:
                x = [self.X[int(_p)][0] for _p in p]
                y = [self.X[int(_p)][1] for _p in p]
                z = [self.X[int(_p)][2] for _p in p]
                axes.plot(x, y, z, style)

    def compute_continuation(self, cluster1, cluster2, THRESHOLD_CONTINUATION, mean_dist, prox_threshold):
        smoothness = self.compute_local_transition(cluster1, cluster2, THRESHOLD_CONTINUATION, mean_dist, prox_threshold)
        # smoothness = self.compute_local_smoothness(p1, p2, radius)

        print(f'---Smoothness: {smoothness}')
        # return (similarity + smoothness) / 2 if similarity else smoothness
        return smoothness

    def compute_max_angle(self, local_points, ref_point):
        max_p1, max_angle1 = local_points[-1]
        for i in reversed(range(len(local_points)-1)):
            angle = self.to_find_angle(self.X[int(ref_point)], self.X[int(max_p1)], self.X[int(local_points[i][0])])
            if angle > max_angle1:
                info = max_p1, local_points[i][0], angle
                break
        else:
            angle = self.to_find_angle(self.X[int(ref_point)], self.X[int(max_p1)], self.X[int(local_points[0][0])])
            info = max_p1, local_points[0][0], angle if angle != 0 else max_angle1
        return info

    def compute_local_transition(self, cluster1, cluster2, THRESHOLD_CONTINUATION, mean_dist, prox_threshold):
        '''
        Compute the density transition and angle transition between two clusters;
        Identify core and boundary cases.
        :param cluster1:
        :param cluster2:
        :return:
        '''
        p1, p2 = self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['reference_points'][
                     cluster1], \
                 self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['reference_points'][
                     cluster2]
        points_dist = np.linalg.norm(self.X[p1] - self.X[p2])
        middle_point = (self.X[p1] + self.X[p2]) / 2

        past_dists = [*self.initial_clusters[cluster1]['past_dists'][-5:],
                      *self.initial_clusters[cluster2]['past_dists'][-5:]]

        size1, size2 = len(self.initial_clusters[cluster1]['data']), \
                       len(self.initial_clusters[cluster2]['data'])
        if size1 > 2:
            compact1 = (self.initial_clusters[cluster1]['past_std'][-1] /
                        np.mean(self.initial_clusters[cluster1]['past_dists']) / np.log(size1)) * size1 / (size1 + size2)
        else:
            compact1 = 0.5
        if size2 > 2:
            compact2 = (self.initial_clusters[cluster2]['past_std'][-1] /
                        np.mean(self.initial_clusters[cluster2]['past_dists']) / np.log(size2)) * size2 / (size1 + size2)
        else:
            compact2 = 0.5

        print("Compact", compact1, compact2)
        compact = compact1 + compact2

        c_past_dists = [max(past_dists), points_dist] if len(past_dists) else [points_dist]
        center_dist = np.linalg.norm(self.initial_clusters[cluster1]['center'] -
                                     self.initial_clusters[cluster2]['center']) / compact

        rate = max(center_dist / points_dist, 3) - 3
        enlarge_rate = 2 / (1 + math.e ** (-rate/20))
        base_length = np.mean(c_past_dists) * enlarge_rate # (points_dist + mean_dist) / 2

        explore_range = [2 + i for i in [0, 0.5, 1]]
        # explore_range=[2, 2.5, prox_threshold]
        # explore_range = [prox_threshold, prox_threshold + 0.5]

        locality_info = [{'radius': r * base_length,
                          'red': {'mean angle': 0,
                                  'std': np.inf,
                                  'max angle': 0,
                                  'N': 0,
                                  'mass': 1,
                                  'local_points': []},
                          'green': {'mean angle': 0,
                                    'std': np.inf,
                                    'max angle': 0,
                                    'N': 0,
                                    'mass': 1,
                                    'local_points': []},
                          'angle_smoothness': 1,
                          'mass_smoothness': 1,
                          'transition_smoothness': 1,
                          'orientation_smoothness': 0.0,
                          'smoothness': 1,
                          'is_boundary': False
                          } for r in explore_range]
        min_radius = None
        surrounded = False
        boundary_cases = []

        for i, radius in enumerate(explore_range):
            _r = radius * base_length
            _local_points1 = self.find_local_points(middle_point, p1, _r, p2)
            _local_points2 = self.find_local_points(middle_point, p2, _r, p1)

            locality_info[i]['red']['local_points'] = _local_points1
            locality_info[i]['green']['local_points'] = _local_points2

            N1, N2 = len(_local_points1) + 1, len(_local_points2) + 1

            locality_info[i]['radius'] = _r
            locality_info[i]['red']['N'] = N1
            locality_info[i]['green']['N'] = N2
            # r_e, r_i, g_i, g_e = self.compute_transition_state(p1, p2, middle_point, N1, N2, _r)

            # angle_smoothness = self.compute_angle_smoothness(_local_points1, _local_points2)
            _local_points1 = self.remove_outliers(_local_points1, middle_point)
            _local_points2 = self.remove_outliers(_local_points2, middle_point)

            if len(_local_points1) > 1 and len(_local_points2) > 1:
                if not surrounded:
                    count2 = sum(
                        [1 if int(p) in self.initial_clusters[cluster1]['data'] else 0 for p in _local_points2[:, 0]])
                    surrounded = count2 >= len(self.initial_clusters[cluster2]['data']) and count2 > N2 / 2

                max_angle_info1 = self.compute_max_angle(_local_points1, p1)
                max_angle_info2 = self.compute_max_angle(_local_points2, p2)
                # compute the local density
                area1, area2 = max_angle_info1[-1], max_angle_info2[-1]
                if min(area1, area2) == 0 and max(area1, area2) <= 0.2: # to handle zero case
                    mass1 = mass2 = 1
                else:
                    mass1, mass2 = N1 * area1, N2 * area2
                locality_info[i]['red']['mass'] = mass1
                locality_info[i]['green']['mass'] = mass2
                locality_info[i]['red']['max angle'] = area1
                locality_info[i]['green']['max angle'] = area2
                locality_info[i]['mass_smoothness'] = min(mass1, mass2) / max(mass1, mass2)

                angle_smoothness = self.compute_angle_transition_2(p1, p2, [max_angle_info1[0],
                                                                            max_angle_info1[1],
                                                                            max_angle_info2[0],
                                                                            max_angle_info2[1]],
                                                                   _local_points1, _local_points2)
                locality_info[i]['angle_smoothness'] = angle_smoothness

                transition_smoothness = self.compute_transition_smoothness(p1, p2, middle_point, N1, N2, _r, radius-1)
                locality_info[i]['transition_smoothness'] = transition_smoothness

            orientation_smoothness = math.sqrt(locality_info[i]['mass_smoothness'] *
                                                       locality_info[i]['angle_smoothness'])
            locality_info[i]['orientation_smoothness'] = orientation_smoothness

            smoothness = min(locality_info[i]['transition_smoothness'] * orientation_smoothness, 1)

            locality_info[i]['smoothness'] = smoothness if not surrounded else 1

            if locality_info[i]['transition_smoothness'] > 2:
                locality_info[i]['is_boundary'] = True
                # boundary_cases.append(locality_info[i]['transition_smoothness'])

            if locality_info[i]['smoothness'] <= THRESHOLD_CONTINUATION:
                min_radius = i

            if i > 0:
                if locality_info[i]['is_boundary'] and locality_info[i-1]['is_boundary']: # boundary case
                    min_radius = i
                    locality_info[i]['smoothness'] = 1
                    break

                if locality_info[i-1]['smoothness'] == locality_info[i]['smoothness'] == 1:
                    min_radius = i
                    break

                if locality_info[i]['mass_smoothness'] / locality_info[i-1]['mass_smoothness'] > np.e:
                    min_radius = i
                    break

        if min_radius is None:
            min_radius = -1

        if len(self.initial_clusters) <= START_PLOT and LOCALITY:  #  and cluster and refe
            # f, axes = plt.subplots(2, 3, figsize=(15, 12))
            fig = plt.figure(constrained_layout=True, figsize=(16, 12))
            gs = GridSpec(3, 4, figure=fig)
            if self.DIMENSION == 2:
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.set_aspect('equal', adjustable='box')
            else:
                ax1 = fig.add_subplot(gs[0, 0], projection='3d')

            self.plot_point(self.X, 'k.', axes=ax1)
            self.plot_point(self.X[p1], 'r*', axes=ax1)
            self.plot_point(self.X[p2], 'g*', axes=ax1)

            self.plot_point(self.initial_clusters[cluster1]['data'], 'r.', axes=ax1)
            self.plot_point(self.initial_clusters[cluster2]['data'], 'g.', axes=ax1)

            text = []
            std_text = ''
            for i in range(1, len(explore_range) + 1):
                if self.DIMENSION == 2:
                    ax = fig.add_subplot(gs[i//4, i % 4])
                else:
                    ax = fig.add_subplot(gs[i // 4, i % 4], projection='3d')

                self.plot_point(self.X, 'k.', axes=ax)
                self.plot_point(self.X[p1], 'r*', axes=ax)
                self.plot_point(self.X[p2], 'g*', axes=ax)

                _local_points1 = locality_info[i-1]['red']['local_points']
                _local_points2 = locality_info[i-1]['green']['local_points']

                self.plot_point([p[0] for p in _local_points1], 'r.', axes=ax)
                self.plot_point([p[0] for p in _local_points2], 'g.', axes=ax)

                if self.DIMENSION == 2:
                    cir = plt.Circle(self.X[p1], locality_info[i - 1]['radius'], color='r', fill=False)
                    cir2 = plt.Circle(self.X[p2], locality_info[i - 1]['radius'], color='g', fill=False)

                    ax.set_aspect('equal', adjustable='box')
                    ax.add_patch(cir)
                    ax.add_patch(cir2)

                text.append([f"{locality_info[i-1]['radius']:.3f}",
                             f"{locality_info[i-1]['smoothness']:.3f}",
                             f"{locality_info[i-1]['angle_smoothness']:.3f}",
                             f"{locality_info[i-1]['mass_smoothness']:.3f}",
                             f"{locality_info[i-1]['red']['N']}",
                             f"{locality_info[i-1]['transition_smoothness']:.3f}",
                             f"{locality_info[i-1]['green']['N']}"])

                std_text += f"Red-Green density: {locality_info[i-1]['red']['mass']:.3f} - " \
                            f"{locality_info[i-1]['red']['max angle']:.3f}, " \
                                f"{locality_info[i-1]['green']['mass']:.3f} - {locality_info[i-1]['green']['max angle']:.3f}\n"
            std_text += f"Enlarge: {enlarge_rate:.3f}, d_point: {points_dist:.3f}; c_dist: {center_dist:.3f}; " \
                        f"adap_prox: {prox_threshold*base_length:.3f}\n"

            ax2 = fig.add_subplot(gs[1, 2:])
            ax2.table(cellText=[['radius', 'smoothness', 'angle', 'mass', 'Red N', 'transition', 'Green N']],
                      loc='bottom',
                      bbox=[0, 0.9, 1, 0.1])

            ax2.table(cellText=text,
                      loc='bottom',
                      bbox=[0, 0.3, 1, 0.6])

            ax2.text(0, 0, std_text + f'clusters: {cluster1}-{cluster2}')
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

            ax3 = fig.add_subplot(gs[2, :2])
            ax3.table(cellText=[['Angle statistics']],
                      loc='bottom',
                      bbox=[0.1, 0.9, 0.2, 0.1]
                      )

            ax3.table(cellText=[['Circle', 'Mean', 'Std', 'Mean', 'Std', 'N', 'Mean', 'Std', 'Exter N', 'Inter N']],
                      loc='bottom',
                      bbox=[0, 0.8, 1, 0.1]
                      )

            ax3.table(cellText=[['Density']],
                      loc='bottom',
                      bbox=[0.3, 0.9, 0.3, 0.1]
                      )
            ax3.table(cellText=[['Distance']],
                      loc='bottom',
                      bbox=[0.6, 0.9, 0.2, 0.1]
                      )
            ax3.table(cellText=[['transition']],
                      loc='bottom',
                      bbox=[0.8, 0.9, 0.2, 0.1]
                      )

            ax3.table(cellText=[['Red', f'{0:.4f}', f"{0:.4f}", f"{0:.4f}",
                                 f"{0:.4f}", f"{locality_info[min_radius]['red']['N']}", f"{0: .4f}",
                                 f"{0:.4f}", 0, 0],
                                ['Green', f'{0:.4f}', f"{0:.4f}", f"{0:.4f}",
                                 f"{0:.4f}", f"{locality_info[min_radius]['green']['N']}", f"{0: .4f}",
                                 f"{0:.4f}", 0, 0],
                                ['', '', '', '', '', f"{0:.4f}", '', '',
                                 f"{0:.4f}", f"{0:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0.6, 1, 0.2]
                      )

            ax3.table(cellText=[[f'Transition smoothness <T1, T2, T3>', f" {0:.4f}, {0:.4f}, {0:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0.5, 1, 0.1]
                      )

            ax3.table(cellText=[['Max angle']],
                      loc='bottom',
                      bbox=[0, 0.3, 0.28, 0.1]
                      )

            ax3.table(cellText=[['Smoothness']],
                      loc='bottom',
                      bbox=[0.28, 0.3, 0.72, 0.1]
                      )

            ax3.table(cellText=[['Angle', 'Inter angle', 'Shape', 'Density', 'Max angle', 'Distance', 'transition']],
                      loc='bottom',
                      bbox=[0, 0.2, 1, 0.1]
                      )

            ax3.table(cellText=[[f"{locality_info[min_radius]['angle_smoothness']:.4f}", f"", f"{locality_info[min_radius]['mass_smoothness']:.4f}",
                                 f"{0:.4f}", f"{locality_info[min_radius]['angle_smoothness']:.4f}", f"{0:.4f}",
                                 f"{locality_info[min_radius]['transition_smoothness']:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0.1, 1, 0.1]
                      )

            ax3.table(cellText=[[f"Mass smoothness (shape, mass, distance): {0:.4f}",
                                 f"Smoothness: {locality_info[min_radius]['smoothness']:.4f}, "
                                 f"{THRESHOLD_CONTINUATION:.4f}, "
                                 f"{0:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0, 1, 0.1]
                      )

            ax3.get_xaxis().set_ticks([])
            ax3.get_yaxis().set_ticks([])

            if SAVE_FIG:
                fig.savefig(f'{self.log_path}/{len(self.initial_clusters)}_mass.png')
            else:
                plt.show()
            plt.close('all')

        return locality_info[min_radius]['smoothness']
        # return 1

    def contain_outliers(self, local_points, middle_point):

        for i in range(len(local_points) - 1):
            theta = self.to_find_angle(middle_point, self.X[int(local_points[-1][0])], self.X[int(local_points[i][0])])
            if theta < local_points[-1][1] / 2:
                return False
        return True

    def compute_local_angle_info(self, local_points, reference_p, middle_point):
        angle_info = []
        n = len(local_points)

        for i in range(n):
            for j in range(i+1, n):
                theta0 = self.to_find_angle(middle_point, self.X[int(local_points[i][0])], self.X[int(local_points[j][0])])
                theta1 = self.to_find_angle(middle_point, self.X[int(local_points[i][0])], self.X[int(reference_p)])
                theta2 = self.to_find_angle(middle_point, self.X[int(local_points[j][0])], self.X[int(reference_p)])
                angle_info.append(np.array([theta0, theta1, theta2, local_points[i][0], local_points[j][0]]))
        return np.array(angle_info)

    def remove_outliers(self, local_points, middle_point):
        n = len(local_points)
        if n > 3:  # to avoid outliers
            if self.contain_outliers(local_points, middle_point):
                n -= 1
                return local_points[:-1]
        return local_points

    def compute_angle_transition_2(self, p1, p2, max_points, local_points1, local_points2):
        def find_smallest_angle(start_p, p, rp, points):
            angle = math.pi
            for v in points:
                theta = self.to_find_angle(start_p, self.X[int(p)], self.X[int(v[0])])
                r_theta = self.to_find_angle(start_p, self.X[int(rp)], self.X[int(v[0])])
                if theta < angle and r_theta >= theta:
                    angle = theta
            return angle

        if len(local_points1) and len(local_points2):
            middle_point = (self.X[p1] + self.X[p2]) / 2

            _p1, _p2, _p3, _p4 = max_points

            transition_angles = [find_smallest_angle(middle_point, _p1, p1, local_points2),
                                 find_smallest_angle(middle_point, _p2, p1, local_points2),
                                 find_smallest_angle(middle_point, _p3, p2, local_points1),
                                 find_smallest_angle(middle_point, _p4, p2, local_points1)]
            angle_transition = [2 * math.fabs(min(angle, math.fabs(angle - math.pi)) - math.pi / 2) / math.pi
                                        for angle in transition_angles]
            angle_transition = np.mean(angle_transition)
            return angle_transition
        else:
            return 1

    def compute_angle_transition(self, local_angle_info1, local_angle_info2, p1, p2, local_points1, local_points2):
        def find_smallest_angle(start_p, p, rp, points):
            angle = math.pi
            for v in points:
                theta = self.to_find_angle(start_p, self.X[int(p)], self.X[int(v[0])])
                r_theta = self.to_find_angle(start_p, self.X[int(rp)], self.X[int(v[0])])
                if theta < angle and r_theta >= theta:
                    angle = theta
            return angle

        if len(local_angle_info1) and len(local_angle_info2):
            middle_point = (self.X[p1] + self.X[p2]) / 2
            local_angle_info1 = sorted(local_angle_info1, key=lambda x: x[0], reverse=True)
            local_angle_info2 = sorted(local_angle_info2, key=lambda x: x[0], reverse=True)

            _p1, _p2, _p3, _p4 = local_angle_info1[0][3], local_angle_info1[0][4], \
                             local_angle_info2[0][3], local_angle_info2[0][4]

            transition_angles = [find_smallest_angle(middle_point, _p1, p1, local_points2),
                                 find_smallest_angle(middle_point, _p2, p1, local_points2),
                                 find_smallest_angle(middle_point, _p3, p2, local_points1),
                                 find_smallest_angle(middle_point, _p4, p2, local_points1)]
            angle_transition = [2 * math.fabs(min(angle, math.fabs(angle - math.pi)) - math.pi / 2) / math.pi
                                        for angle in transition_angles]
            angle_transition = np.mean(angle_transition)
            return angle_transition
            # theta1 = self.to_find_angle(middle_point, _p1, _p3)
            # theta2 = self.to_find_angle(middle_point, _p1, _p4)
            #
            # if theta1 < theta2:
            #     theta = self.to_find_angle(middle_point, _p2, _p3)
            #     if theta < theta1:
            #         left_angle, right_angle = math.pi - local_angle_info1[0][2] - local_angle_info2[0][1], \
            #                                   math.pi - local_angle_info1[0][1] - local_angle_info2[0][2]
            #     else:
            #         left_angle, right_angle = math.pi - local_angle_info1[0][1] - local_angle_info2[0][1], \
            #                                   math.pi - local_angle_info1[0][2] - local_angle_info2[0][2]
            # else:
            #     theta = self.to_find_angle(middle_point, _p2, _p4)
            #     if theta < theta1:
            #         left_angle, right_angle = math.pi - local_angle_info1[0][2] - local_angle_info2[0][2], \
            #                                   math.pi - local_angle_info1[0][1] - local_angle_info2[0][1]
            #     else:
            #         left_angle, right_angle = math.pi - local_angle_info1[0][1] - local_angle_info2[0][2], \
            #                                   math.pi - local_angle_info1[0][2] - local_angle_info2[0][1]
            #
            # # left_angle, right_angle = math.pi - angle1 - angle2, math.pi - angle3 - angle4
            # l1, r1 = (2 * math.fabs(min(left_angle, math.fabs(left_angle - math.pi)) - math.pi / 2) / math.pi), \
            #          (2 * math.fabs(min(right_angle, math.fabs(right_angle - math.pi)) - math.pi / 2) / math.pi)
            # angle_transition = (l1 + r1) / 2
            # return angle_transition
        else:
            return 1

    def compute_shape_smoothness(self, local_points1, local_points2):
        N1, N2 = len(local_points1) + 1, len(local_points2) + 1
        if N1 >= 4 and N2 >= 4:
            density_Ratio = min(N1, N2) / max(N1, N2) if max(N1, N2) != 0 else 0

            std_angle1 = np.std([p[1] for p in local_points1])
            std_angle2 = np.std([p[1] for p in local_points2])

            if std_angle1 < 0.05 and std_angle2 < 0.05:  # this is to deal with if two stds are too small
                shape_smoothness = 0.5
            else:
                shape_smoothness = (min(std_angle1, std_angle2) /
                                    max(std_angle1, std_angle2)) * density_Ratio \
                    if min(std_angle1, std_angle2) != 0 else 0.1
        else:
            shape_smoothness = 1
        return shape_smoothness

    def compute_local_shape_smoothness(self, local_angle_info1, local_angle_info2, density_Ratio):
        if len(local_angle_info1) >= 2 and len(local_angle_info2) >= 2:
            rotation_std1 = np.std(local_angle_info1[:, 0])
            rotation_std2 = np.std(local_angle_info2[:, 0])

            polar_std1 = np.std(local_angle_info1[:, 1:3])
            polar_std2 = np.std(local_angle_info2[:, 1:3])

            if rotation_std1 < 0.1 and rotation_std2 < 0.1 and polar_std1 < 0.1 and polar_std2 < 0.1:
                # this is to deal with if two types of stds are too small
                shape_smoothness = 0.5
            else:
                rotation, polar = (min(rotation_std1, rotation_std2) / max(rotation_std1, rotation_std2)) * \
                                  density_Ratio if max(rotation_std1, rotation_std2) != 0 else 1, \
                                  (min(polar_std1, polar_std2) / max(polar_std1, polar_std2)) * \
                                  density_Ratio if max(polar_std1, polar_std2) != 0 else 1
                shape_smoothness = (rotation + polar) / 2

        else:
            shape_smoothness = 1
        return shape_smoothness

    def find_near_points(self, p1, p2, cluster1, cluster2):
        middle_point = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2)
        local_points1 = self.get_points_wthin_radius(p1, radius)
        local_points2 = self.get_points_wthin_radius(p2, radius)
        intersections = np.array([x for x in set(tuple(x) for x in local_points1) &
                                  set(tuple(x) for x in local_points2)])
        intersections = sorted([[p, np.linalg.norm(p - middle_point)] for p in intersections], key=lambda x: x[1])
        return [p[0] for p in intersections if not self.is_point_exist(p[0], self.initial_clusters[cluster1]['data']) and
                not self.is_point_exist(p[0], self.initial_clusters[cluster2]['data'])]

    def compute_transition_smoothness(self, p1, p2, middle_point, N1, N2, radius, r_rate, points_set=None):
        r_e, r_i, g_i, g_e = self.compute_transition_state(p1, p2, middle_point, N1, N2, radius, points_set=points_set)
        if r_e == 0 or g_e == 0:
            transition_smoothness = max(2, (max(N1, N2) / min(N1, N2)) / r_rate)
        elif ((r_i < g_i or r_i < g_e) and r_i < r_e) or (g_i < g_e and (g_i < r_i or g_i < r_e)):
            transition_smoothness = min(min(r_i, g_i) / max(g_i, g_e), min(r_i, g_i) / max(r_i, r_e))  # if min(NN) != 0 else 0.5
            # transition_smoothness = 0.2
        else:
            if r_e / max([g_i, g_e, r_i]) <= 0.1 or g_e / max([g_i, r_e, r_i]) <= 0.1:
                transition_smoothness = 1.5
            else:
                transition_smoothness = 1
        return transition_smoothness

    def compute_local_smoothness(self, p1, p2, radius):
        middle_point = (p1 + p2) / 2
        local_points1 = self.find_local_points(middle_point, p1, radius, p2)
        local_points2 = self.find_local_points(middle_point, p2, radius, p1)

        N1, N2 = len(local_points1) + 1, len(local_points2) + 1
        density_Ratio = min(N1, N2) / max(N1, N2) if max(N1, N2) != 0 else 0

        # compute angle statistics
        internal_ratio, external_ratio = 0, 0
        T1, T2, T3, T3_2 = 0, 0, 0, 0
        transition_smoothness = 1

        left_angle, right_angle = 0, 0
        mean_angle1, mean_angle2, std_angle1, std_angle2 = 0, 0, 0, 0
        mean1, mean2, std1, std2 = 0, 0, 0, 0
        dist_mean1, dist_mean2, dist_std1, dist_std2 = 0, 0, 0, 0
        r_e, g_e, r_i, g_i = 0, 0, 0, 0
        shape_smoothness, angle_smoothness, \
        distance_smoothness = 1, 1, 1
        density_smoothness, new_density = 1, 1

        if len(local_points1) and len(local_points2):
            # compute max angle
            local_angle_info1, local_angle_info2 = self.compute_local_angle_info(local_points1, p1, middle_point), \
                                                   self.compute_local_angle_info(local_points2, p2, middle_point)

            angle_smoothness = self.compute_angle_transition(local_angle_info1, local_angle_info2, middle_point)

            # compute transition smoothness
            transition_smoothness = self.compute_transition_smoothness(p1, p2, middle_point, N1, N2, radius)

            # shape_smoothness = self.compute_shape_smoothness(local_points1, local_points2)

            shape_smoothness = self.compute_local_shape_smoothness(local_angle_info1, local_angle_info2, density_Ratio)
            # print(f"--Shape_smoothness: {shape_smoothness}; Local shape: {local_shape}")
            # print(f"--Formulated local shape: {l_s_s}")

            r_e, r_i, g_i, g_e = self.compute_transition_state(p1, p2, middle_point, N1, N2, radius)

            new_density = math.sqrt(density_smoothness * distance_smoothness)

            print('-----Is boundary: ', angle_smoothness, density_smoothness)

        orientation_smoothness = math.sqrt(shape_smoothness * angle_smoothness)

        # if new_density > orientation_smoothness:
        #     smoothness = min(
        #         transition_smoothness * orientation_smoothness, 1)
        # else:
        #     smoothness = min(
        #         transition_smoothness * (new_density + orientation_smoothness) / 2, 1)

        # if angle_smoothness > shape_smoothness:
        #     smoothness = shape_smoothness * (1 - angle_smoothness) + angle_smoothness * angle_smoothness
        # else:
        #     smoothness = shape_smoothness * shape_smoothness + angle_smoothness * (1 - shape_smoothness)
        smoothness = min(transition_smoothness * orientation_smoothness, 1)

        if len(self.initial_clusters) <= START_PLOT and CONTINUATION_DEBUG:  # len(self.initial_clusters) <= 200

            if self.DIMENSION == 2:
                f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
            else:
                f = plt.figure(figsize=(8, 10))
                ax1 = f.add_subplot(2, 1, 1, projection='3d')
                ax2 = f.add_subplot(2, 1, 2)

            self.plot_point(self.X, 'k.', axes=ax1)
            self.plot_point(p1, 'r*', axes=ax1)
            self.plot_point(p2, 'g*', axes=ax1)

            self.plot_point([p[0] for p in local_points1], 'r.', axes=ax1)
            self.plot_point([p[0] for p in local_points2], 'g.', axes=ax1)

            # self.plot_point(self.initial_clusters[cluster1]['data'], 'r.')
            # self.plot_point(self.initial_clusters[cluster2]['data'], 'g.')

            # self.plot_point(self.X[n_ps[-1][0]], 'b.')
            # self.plot_point(self.X[n_rps[0][0]], 'b.')

            # self.plot_point(self.X[n_ps[0][0]], 'y.')
            # self.plot_point(self.X[n_rps[-1][0]], 'y.')

            # plt.show()

            if self.DIMENSION == 2:
                ax1.set_aspect('equal', adjustable='box')
                cir = plt.Circle(p1, radius, color='r', fill=False)
                cir2 = plt.Circle(p2, radius, color='g', fill=False)
                ax1.add_patch(cir)
                ax1.add_patch(cir2)

            ax2.table(cellText=[['Angle statistics']],
                      loc='bottom',
                      bbox=[0.1, 0.9, 0.2, 0.1]
                      )

            ax2.table(cellText=[['Circle', 'Mean', 'Std', 'Mean', 'Std', 'N', 'Mean', 'Std', 'Exter N', 'Inter N']],
                      loc='bottom',
                      bbox=[0, 0.8, 1, 0.1]
                      )

            ax2.table(cellText=[['Density']],
                      loc='bottom',
                      bbox=[0.3, 0.9, 0.3, 0.1]
                      )
            ax2.table(cellText=[['Distance']],
                      loc='bottom',
                      bbox=[0.6, 0.9, 0.2, 0.1]
                      )
            ax2.table(cellText=[['transition']],
                      loc='bottom',
                      bbox=[0.8, 0.9, 0.2, 0.1]
                      )

            ax2.table(cellText=[['Red',f'{mean_angle1:.4f}', f"{std_angle1:.4f}", f"{mean1:.4f}",
                                 f"{std1:.4f}", f"{N1}" , f"{dist_mean1: .4f}",
                                 f"{dist_std1:.4f}",  r_e, r_i],
                                ['Green',f'{mean_angle2:.4f}', f"{std_angle2:.4f}", f"{mean2:.4f}",
                                 f"{std2:.4f}", f"{N2}" , f"{dist_mean2: .4f}",
                                 f"{dist_std2:.4f}",  g_e, g_i],
                                ['', '', '', '', '', f"{density_Ratio:.4f}", '', '',
                                 f"{external_ratio:.4f}", f"{internal_ratio:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0.6, 1, 0.2]
                      )

            ax2.table(cellText=[[f'Transition smoothness <T1, T2, T3>', f" {T1:.4f}, {T2:.4f}, {T3:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0.5, 1, 0.1]
                      )

            ax2.table(cellText=[['Max angle']],
                      loc='bottom',
                      bbox=[0, 0.3, 0.28, 0.1]
                      )

            ax2.table(cellText=[['Smoothness']],
                      loc='bottom',
                      bbox=[0.28, 0.3, 0.72, 0.1]
                      )

            ax2.table(cellText=[['Angle', 'Inter angle', 'Shape', 'Density', 'Max angle', 'Distance', 'transition']],
                      loc='bottom',
                      bbox=[0, 0.2, 1, 0.1]
                      )

            ax2.table(cellText=[[f"{angle_smoothness:.4f}", f"", f"{shape_smoothness:.4f}",
                        f"{density_smoothness:.4f}", f"{angle_smoothness:.4f}", f"{distance_smoothness:.4f}",
                        f"{transition_smoothness:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0.1, 1, 0.1]
                      )

            ax2.table(cellText=[[f"Density smoothness (shape, density, distance): {new_density:.4f}",
                                 f"Smoothness: {smoothness:.4f}, "
                                 f"{(new_density + angle_smoothness) / 2:.4f}, "
                                 f"{(new_density + angle_smoothness + transition_smoothness) / 3:.4f}"]],
                      loc='bottom',
                      bbox=[0, 0, 1, 0.1]
                      )

            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])
            if SAVE_FIG:
                f.savefig(f'{self.log_path}/{len(self.initial_clusters)}_circles.png')
            else:
                plt.show()
            plt.close('all')

        return smoothness

    def compute_transition_state(self, p1, p2, middle_point, N1, N2, radius, points_set=None):
        all_local_points1 = self.find_local_points(middle_point, p1, radius, p2, all=True, points_set=points_set)
        all_local_points2 = self.find_local_points(middle_point, p2, radius, p1, all=True, points_set=points_set)

        g_i = len(all_local_points1) + 1 - N1
        r_i = len(all_local_points2) + 1 - N2

        return N1 - r_i, r_i, g_i, N2 - g_i


    def find_common_points(self, group1, group2):
        intersections = np.array([x for x in (tuple(x) for x in group1) if x in (tuple(x) for x in group2)])

        return intersections

    def compute_distance_smoothness(self, cluster1, cluster2):
        p1, p2 = self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['reference_points'][
                     cluster1], \
                 self.clusters_dist[frozenset((cluster1, cluster2))]['distance_info']['near_dist']['reference_points'][
                     cluster2]
        points_dist = np.linalg.norm(p1 - p2)
        radius = 2.5 * points_dist

        middle_point = (p1 + p2) / 2

        local_points1 = self.find_local_points(middle_point, p1, radius, p2)
        local_points2 = self.find_local_points(middle_point, p2, radius, p1)

        if len(local_points1):
            dist_p2_local_points1 = sum([np.linalg.norm(p[0] - p2) for p in local_points1]) / \
                                    len(local_points1)
        else:
            dist_p2_local_points1 = points_dist

        if len(local_points2):
            dist_p1_local_points2 = sum([np.linalg.norm(p[0] - p1) for p in local_points2]) / \
                                    len(local_points2)
        else:
            dist_p1_local_points2 = points_dist

        dist_smoothness =  min(dist_p2_local_points1, dist_p1_local_points2) / \
              max(dist_p2_local_points1, dist_p1_local_points2)

        print('----Distance smoothness', dist_p2_local_points1, dist_p1_local_points2, dist_smoothness)

        return dist_smoothness

    def find_local_points(self, middle_point, point, radius, exclude_point, cluster=None, all=False, points_set=None):
        local_points = []
        for p in self.get_points_wthin_radius(point, radius, cluster):
            # angle = self.to_find_clockwise_angle(middle_point, point, p)
            angle = self.to_find_angle(middle_point, self.X[int(point)], self.X[int(p)])
            # angle = math.pi - self.to_find_angle(point, middle_point, p)
            if not all:
                if (angle <= math.pi / 2 or angle >= 3 * math.pi / 2) and p != exclude_point:
                    # _angle = self.to_find_clockwise_angle(exclude_point, point, p)
                    # _angle = _angle if _angle <= math.pi / 2 else _angle - 2 * math.pi
                    # local_points.append([p, _angle])
                    _angle = angle if angle <= math.pi / 2 else angle - 2 * math.pi
                    local_points.append([p, _angle])
            else:
                angle = angle if angle <= math.pi else angle - 2 * math.pi
                local_points.append([p, angle])

        return np.array(sorted(local_points, key=lambda x: x[1]))

    @staticmethod
    def to_find_clockwise_angle(start, left_p, right_p):
        v1 = [left_p[0] - start[0], left_p[1] - start[1]]
        v2 = [right_p[0] - start[0], right_p[1] - start[1]]

        theta = - math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
        angle = round(theta, 4) if math.copysign(1, theta) >= 0 else round(2 * math.pi + theta, 4)
        return angle

    @staticmethod
    def to_find_angle(start, left_p, right_p):
        v1 = left_p - start
        v2 = right_p - start

        dot_p = np.dot(v1, v2)
        if round(np.linalg.norm(v1) * np.linalg.norm(v2), 4) == 0:
            theta = 0
        else:
            theta = np.arccos(round(dot_p / (np.linalg.norm(v1) * np.linalg.norm(v2)), 4))
        return theta

    def compute_inter_clusters_distances(self, clusters):
        dists = {}
        list_clusters = list(clusters.items())
        for i in range(len(list_clusters)):
            for j in range(i + 1, len(list_clusters)):
                dists[frozenset([list_clusters[i][0],
                                 list_clusters[j][0]])] = self.compute_2_clusters_dist(list_clusters[i][1],
                                                                            list_clusters[j][1])
        return dists

    def compute_inter_points_distances(self, points):
        dists = {}
        list_clusters = list(points.items())
        for i in range(len(list_clusters)):
            for j in range(i + 1, len(list_clusters)):
                _d = self.compute_2_clusters_dist(list_clusters[i][1], list_clusters[j][1])
                if list_clusters[i][0] in dists.keys():
                    dists[list_clusters[i][0]].append([list_clusters[j][0], _d])
                else:
                    dists[list_clusters[i][0]] = [[list_clusters[j][0], _d]]
                if list_clusters[j][0] in dists.keys():
                    dists[list_clusters[j][0]].append([list_clusters[i][0], _d])
                else:
                    dists[list_clusters[j][0]] = [[list_clusters[i][0], _d]]
        for k, v in dists.items():
            dists[k] = list(sorted(v, key=lambda x: x[1]))

        return dists

    def get_mean_dist_4_inter_clusters(self):
        sum = 0
        for k, v in self.clusters_dist.items():
            sum += v['distance']
        return sum/len(self.clusters_dist)

    def get_min_dist_4_inter_clusters(self):
        values = sorted(self.clusters_dist.items(), key=lambda item: item[1]['distance'])
        for v in values:
            if v[1] != 0:
                return v[1]

    def get_mean_dist_4_intra_clusters(self):
        sum = 0
        for k, v in self.initial_clusters.items():
            sum += v['mean_dist']
        return sum/len(self.initial_clusters) if sum != 0 else 0.001

    def closest_points_btn_2_clusters(self, cluster1, cluster2):
        closest_points = None
        min_dist = np.inf
        for p1 in self.initial_clusters[cluster1]['data']:
            for p2 in self.initial_clusters[cluster2]['data']:
                dist = np.linalg.norm(self.X[int(p1)] - self.X[int(p2)])
                if dist < min_dist:
                    min_dist = dist
                    closest_points = [p1, p2]
        return closest_points

    def rank_points_btn_2_clusters(self, cluster1, cluster2):
        rank_points = []
        for p1 in self.initial_clusters[cluster1]['data']:
            for p2 in self.initial_clusters[cluster2]['data']:
                dist = np.linalg.norm(p1 - p2)
                rank_points.append([[p1, p2], dist])
        return list(sorted(rank_points, key=lambda x: x[1]))

    def get_nearest_k_points(self, k, point):
        id = int(np.where((self.X == point).all(axis=1))[0][0])
        return self.points_dist[id][:k]

    def get_points_wthin_radius(self, point, radius, cluster=None, points_set=None):
        points = []
        points_set = self.points_dist[point]

        for p in points_set:
            if p[1] > radius:
                break

            if cluster is not None:
                if np.where((self.initial_clusters[cluster]['data'] == p[0]).all(axis=1))[0].shape == (0,):
                    continue
            points.append(p[0])

        return points

    def is_point_exist(self, point, group):
        if len(group):
            return not np.where((group == point).all(axis=1))[0].shape == (0,)
        else:
            return False

    def predict(self, X):
        pass

    def plot(self, withline=False):
        # Y_hat = []
        # X_hat = []
        # [[[Y_hat.append(int(v['label'])), X_hat.append(x)] for x in v['data']]
        #         for k, v in self.initial_clusters.items()]
        #
        # yhat, xhat = np.array(Y_hat), np.array(X_hat)
        # assign a cluster to each example
        # yhat = model.predict(X)
        # retrieve unique clusters
        # clusters = np.unique(yhat)
        f = plt.figure(figsize=(12, 12))
        if self.DIMENSION == 2:
            ax1 = f.add_subplot(2, 2, 1)
        else:
            ax1 = f.add_subplot(2, 2, 1, projection='3d')
        ax2 = f.add_subplot(2, 2, 2)
        ax3 = f.add_subplot(2, 2, 3)
        ax4 = f.add_subplot(2, 2, 4)
        # f, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='3d'))

        # create scatter plot for samples from each cluster
        # for cluster in clusters:
        #     # get row indexes for samples with this cluster
        #     row_ix = np.where(yhat == cluster)
        #     # create scatter of these samples
        #     axes[0, 0].scatter(xhat[row_ix, 0], xhat[row_ix, 1])

        for k, cluster in self.initial_clusters.items():
            if self.DIMENSION == 2:
                scatter = ax1.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1])
                ax1.text(self.X[cluster['data'][0]][0], self.X[cluster['data'][0]][1], k)
                if withline:
                    for track in cluster['traces']:
                        # axes[0, 0].plot(np.array(track)[:, 0], np.array(track)[:, 1],
                        #                 color=scatter.get_facecolor()[0], linestyle='-')
                        ax1.plot(self.X[list(track)][:, 0], self.X[list(track)][:, 1], 'k-')
            else:
                scatter = ax1.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1],
                                      self.X[cluster['data']][:, 2])
                # ax1.text(cluster['data'][0][0], cluster['data'][0][1], cluster['data'][0][2], k)
                # if withline:
                #     for track in cluster['traces']:
                #         # axes[0, 0].plot(np.array(track)[:, 0], np.array(track)[:, 1],
                #         #                 color=scatter.get_facecolor()[0], linestyle='-')
                #         ax1.plot(np.array(track)[:, 0], np.array(track)[:, 1], np.array(track)[:, 2], 'k-')

        # show the plot
        # plt.show()

        for k, v in self.initial_clusters.items():
            if len(v['merging_dists']) <= 1:
                continue
            ratio = [v['merging_dists'][i] / (sum(v['merging_dists'][: i]) / i) for i in range(1, len(v['merging_dists']))]
            ax2.plot(range(len(v['merging_dists'])), v['merging_dists'], '--.')
            ax2.plot(range(1, len(v['merging_dists'])), ratio, ':')

            ratio2 = [(v['past_densities'][i-1]) / v['past_densities'][i] for i in range(1, len(v['past_densities']))]
            ax3.plot(range(len(v['past_densities'])), v['past_densities'], '--.')
            ax3.plot(range(1, len(v['past_densities'])), ratio2, ':')

            # ratio3 = [v['past_std'][i] / (sum(v['past_std'][: i]) / i) if (sum(v['past_std'][: i]) / i) > 0.01 else 0
            #           for i in range(2, len(v['past_std']))]
            # ratio3 = [v['past_std'][i] / v['past_std'][i-1] if v['past_std'][i-1] > 0.01 else 0
            #           for i in range(1, len(v['past_std']))]
            ax4.plot(range(len(v['past_std'])), v['past_std'], '--.')
            # axes[1, 1].plot(range(1, len(v['past_std'])), ratio3, ':')

        plt.savefig(f'{self.log_path}/{len(self.initial_clusters)}.png')
        plt.close('all')

    # def plot_clusters(self, ax=False, withline=False):
    #
    #     for k, cluster in self.initial_clusters.items():
    #         if self.DIMENSION == 2:
    #             scatter = ax1.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1])
    #             ax1.text(self.X[cluster['data'][0]][0], self.X[cluster['data'][0]][1], k)
    #             if withline:
    #                 for track in cluster['traces']:
    #                     # axes[0, 0].plot(np.array(track)[:, 0], np.array(track)[:, 1],
    #                     #                 color=scatter.get_facecolor()[0], linestyle='-')
    #                     ax1.plot(self.X[list(track)][:, 0], self.X[list(track)][:, 1], 'k-')
    #         else:
    #             scatter = ax1.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1],
    #                                   self.X[cluster['data']][:, 2])

    def plot_clusters_info(self):
        plt.clf()
        for k, v in self.initial_clusters.items():
            if len(v['merging_dists']) <= 1:
                continue
            ratio = [v['merging_dists'][i]/v['merging_dists'][i-1] for i in range(1, len(v['merging_dists']))]
            plt.plot(range(len(v['merging_dists'])), v['merging_dists'], '--.')
            plt.plot(range(1, len(v['merging_dists'])), ratio, ':')
        # plt.show()
        plt.savefig(f'{self.log_path}/{len(self.initial_clusters)}_merging_dists.png')

    def save_clusters(self):
        Y_hat = []
        X_hat = []
        [[[Y_hat.append(int(v['label'])), X_hat.append(self.X[x])] for x in v['data']]
         for k, v in self.initial_clusters.items()]
        yhat, xhat = np.array(Y_hat), np.array(X_hat)
        res = np.column_stack((xhat, yhat))
        np.savetxt(f'{self.log_path}/{len(self.initial_clusters)}.out', res)
        # load_clustering_data(f'{self.log_path}/{len(self.initial_clusters)}.out')