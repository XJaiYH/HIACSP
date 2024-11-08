# LD means use low density objects
import sys
sys.path.append('../../')
import argparse
import heapq
import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
from sklearn.metrics import adjusted_mutual_info_score as AMI, normalized_mutual_info_score as NMI
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.neighbors import KDTree
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo
import mycode.util.dataloader as dataloader
from mycode.util.DPC import DPC
from mycode.util.evaluate import calculate_metric
from mycode.util.visualization import plotCluster, plotKNN
import warnings
warnings.filterwarnings('ignore')

def load_data(args):
    ty = args.data_name.split('.')[-1]
    # ionosphere = fetch_ucirepo(id=52)
    # read_data = ionosphere.data.features.values
    # label = ionosphere.data.targets.values
    if args.load_data_local:
        read_data, label = dataloader.load_dataset(args, ty)
    else:
        read_data, label = dataloader.load_dataset_from_UCI(args.data_id)
    for j in range(read_data.shape[1]):
        max_ = max(read_data[:, j])
        min_ = min(read_data[:, j])
        if max_ == min_:
            continue
        for i in range(read_data.shape[0]):
            read_data[i][j] = (read_data[i][j] - min_) / (max_ - min_)
    return read_data, label

class HIACSP():
    def __init__(self, args, data, label):
        self.K = args.K
        self.data_name = args.data_name
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.ori_data = data
        self.label = label
        self.res_data = self.ori_data.copy()
        self.Number = self.ori_data.shape[0]
        self.Dim = self.ori_data.shape[1]
        self.dis_variation_mean = 0# 两次迭代间的平均移动距离变化
        self.knn_graph = None
        self.knn_euclidean_dis = None
        self.knn_weight_graph = None
        self.knn = None # the last k nearest neighbors
        self.influence_vec = None
        self.influence_norm = None
        self.co_neighbor = np.zeros((self.Number, self.K), dtype=int)
        self.mutual_neighbor = np.zeros((self.Number, self.K), dtype=int)
        self.over_influence_mean = []
        self.tree = KDTree(self.res_data)
        knn_euclidean_dis, _ = self.tree.query(self.res_data, max(self.K + 1, round(self.Number * 0.015)+1))
        area = np.mean(knn_euclidean_dis[:, round(self.Number * 0.015)])
        self.ori_density = self.tree.query_radius(self.res_data, area, count_only=True)
        self.density = None

    def generate_knn_graph(self):
        self.tree = KDTree(self.res_data)
        knn_euclidean_dis, knn_graph = self.tree.query(self.res_data, max(self.K + 1, round(self.Number * 0.015)+1))#tree.query(self.res_data, self.K + 1)
        return knn_graph[:, 1:], knn_euclidean_dis[:, 1:]

    def calculate_weight4graph(self):
        # calculate mutual nearest neighbor and co-nearest neighbors
        self.mutual_neighbor *= 0
        for item in range(self.Number):
            for idx, neighbor in enumerate(self.knn_graph[item, :self.K]):
                item_nn = self.knn_graph[item, :self.K]
                neighbor_nn = self.knn_graph[neighbor, :self.K]
                neighbor_map = {}
                for i in range(self.K):
                    neighbor_map[item_nn[i]] = neighbor_map.get(item_nn[i], 0) + 1
                    neighbor_map[neighbor_nn[i]] = neighbor_map.get(neighbor_nn[i], 0) + 1
                vals = np.array(list(neighbor_map.values()))
                # print(vals, type(vals))
                tmp_count = np.sum(vals > 1)
                self.co_neighbor[item, idx] = tmp_count / (2 * self.K - tmp_count)
                if neighbor_map.get(item, 0) == 1 and neighbor_map.get(neighbor, 0) == 1:
                    self.mutual_neighbor[item, idx] = 1

        # calculate edge weight
        knn_weight_graph = np.zeros((self.Number, self.K))
        for item in range(self.Number):
            for idx in range(self.K):
                if self.mutual_neighbor[item, idx] == 0:
                    # knn_weight_graph[item, idx] = np.inf
                    continue
                else:
                    # continue
                    knn_weight_graph[item, idx] = self.knn_euclidean_dis[item, idx] / (self.co_neighbor[item, idx] + 0.5)

        # knn_weight_graph = np.power(knn_weight_graph, 4)
        knn_weight_graph = np.power(knn_weight_graph, args.p)
        return knn_weight_graph

    def find_knn_by_graph(self):
        knn = -1 * np.ones((self.Number, self.K), dtype=int)
        for item in range(self.Number):
            mark = np.zeros((self.Number, ), dtype=int)
            distances = {node: self.knn_weight_graph[item, t] for t, node in enumerate(self.knn_graph[item, :self.K])}
            distances[item] = 0
            mark[item] = 1
            count = 0
            while count < self.K:
                mini = np.inf
                node = -1
                # find the nearest node of current_node
                for key, val in distances.items():
                    if mini > val and mark[key] == 0:
                        mini = val
                        node = key
                    else:
                        continue
                if node == -1:
                    break
                mark[node] = 1
                knn[item, count] = node
                count += 1

                for i, adj_node in enumerate(self.knn_graph[node, :self.K]):
                    if mark[adj_node] == 0:
                        dis = self.knn_weight_graph[node, i] + distances[node]
                        if distances.get(adj_node) is None:
                            distances[adj_node] = dis
                        elif distances[adj_node] > dis:
                            distances[adj_node] = dis
        return knn

    def plot_influence(self):
        fig, ax = plt.subplots()
        plt.scatter(np.arange(self.Number), self.influence_norm)
        plt.show()

    def calculate_influence(self):
        influence_vec = np.zeros((self.Number, self.Dim))
        for item in range(self.Number):
            j = 0
            while j < self.K and self.knn[item, j] != -1:
                vec = self.res_data[self.knn[item, j]] - self.res_data[item]
                vec_norm = np.linalg.norm(vec)
                if vec_norm == 0:
                    j += 1
                    continue
                unit_vec = vec / vec_norm
                influence_vec[item] += unit_vec
                j += 1
            influence_vec[item] /= self.K
        influence_norm = np.linalg.norm(influence_vec, axis=1)
        return influence_vec, influence_norm

    def multiplier(self, influence, mean):
        # not use
        idx = np.argwhere(influence > mean).reshape(-1, )
        mean_dis = {}
        max_dis = 0
        for item in idx:
            j = 0
            while j < self.K and self.knn[i, j] != -1:
                mean_dis[item] = mean_dis.get(item, 0) + np.sqrt(np.sum(np.square(
                    self.res_data[item] - self.res_data[self.knn[item, j]])))
                j += 1
            mean_dis[item] = mean_dis.get(item, 0) / max(j, 1)
            if max_dis < mean_dis[item]:
                max_dis = mean_dis[item]
        for item in idx:
            mean_dis[item] /= max_dis
        return mean_dis

    def shrink(self, T):
        self.knn_graph, self.knn_euclidean_dis = self.generate_knn_graph()
        self.knn_weight_graph = self.calculate_weight4graph()
        self.knn = self.find_knn_by_graph()
        self.influence_vec, self.influence_norm = self.calculate_influence()
        self.influence_norm = self.influence_norm.reshape(-1, )
        self.over_influence_mean = []

        area = np.mean(self.knn_euclidean_dis[:, round(self.Number * 0.015) - 1])
        self.density = self.tree.query_radius(self.res_data, area, count_only=True)
        # plotKNN(hiacsp.res_data, hiacsp.label, 'test', 20, hiacsp.knn[:, :self.K])
        # plotKNN(hiacsp.res_data, hiacsp.label, 'test', 75, hiacsp.knn[:, :self.K])
        self.influence_norm = self.density
        threshold = np.mean(self.influence_norm)

        self.over_influence_mean = []
        beta = self.res_data.copy()
        for i in range(self.Number):
            if self.influence_norm[i] >= threshold:
                continue
            self.over_influence_mean.append(i)
            displacement = np.zeros(self.res_data.shape[1], dtype=np.float32)
            max_dis = 0
            j = 0
            while j < self.K and self.knn[i, j] != -1:
                dis_2 = np.sum(np.square(self.res_data[self.knn[i, j]] - self.res_data[i]))
                if dis_2 == 0:
                    j += 1
                    continue
                ff = (self.res_data[self.knn[i, j]] - self.res_data[i])
                displacement += ff / np.linalg.norm(ff)
                if max_dis < np.sqrt(dis_2):
                    max_dis = np.sqrt(dis_2)
                j += 1
            tmp_dis = np.linalg.norm(displacement)
            if tmp_dis != 0:
                displacement = displacement * max_dis / tmp_dis
            beta[i] = self.res_data[i] + displacement * T
        if args.visualization:
            plotCluster(self.res_data, self.label, 'test', args, self.over_influence_mean)
        self.res_data = beta.copy()
        return self.res_data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/',
                        help='the director of the original dataset saved')
    parser.add_argument('--save_dir', type=str, default='../../result/', help='the director of results saved')
    parser.add_argument('--data_name', type=str, default='aggregation.txt', help='dataset\'s name')
    parser.add_argument('--ratio', type=int, default=0,
                        help='It is only used to verify the robustness of HIACSP\'s performance under'
                             'different overlap rate, the value of ratio can be selected in'
                             ' [10, 20, 30, 40, 50], datasets are R15, Gauss17, Two, and A1')
    parser.add_argument('--load_data_local', action="store_false",
                        help='Default value is True. If use local dataset, the value should be set as True, otherwise setting as False')
    parser.add_argument('--data_id', type=int, default=78,
                        help='If you want to download dataset from uci, please set the data_id of the dataset you want')
    parser.add_argument('--with_label', action="store_false",
                        help='Default value is True. If the dataset contains labels in the last column, set this value as True. Otherwise, '
                             'this software will load labels from local file which name is '
                             '"args.data_dir/args.data_name/args.data_name-label.txt"')
    parser.add_argument('--visualization', action="store_true",
                        help='Whether to visualization the intermediate results, default value is False')
    parser.add_argument('--K', type=int, default=20, help='Parameter K, range from 5 to 30')
    parser.add_argument('--D', type=int, default=10, help='Iterations')
    parser.add_argument('--T', type=float, default=1., help='Parameter T, range from 0.2 to 1.')
    parser.add_argument('--verbose', action="store_true",
                        help='Whether to print the process information, default value is False')
    parser.add_argument('--p', type=int, default=4, help='hyper-parameter p')
    prompt = '【System Information: 】'
    args = parser.parse_args()

    k_set = [20]
    T_set = [1.]

    algo_name = 'HIACSP-LD'
    data, label = load_data(args)
    clusters = len(set(label))
    km = KMeans(n_clusters=clusters).fit(data)
    print("KM NMI ori: ", AMI(label, km.labels_, average_method='max'))
    agg = AgglomerativeClustering(n_clusters=clusters).fit(data)
    print("AGG NMI ori: ", AMI(label, agg.labels_, average_method='max'))
    dpc = DPC(n_clusters=clusters, center='auto')
    dpc_res = dpc.fit_predict(data)
    print("DPC NMI ori: ", AMI(label, dpc_res, average_method='max'))

    with tqdm(total=len(k_set) * len(T_set) * args.D) as tbar:
        for kk in k_set:
            for tt in T_set:
                args.K = kk
                args.T = tt
                start = time.time()
                hiacsp = HIACSP(args, data.copy(), label)
                clusters = len(set(hiacsp.label))
                plotCluster(hiacsp.ori_data, dpc_res, 'test', args)

                for i in range(args.D):
                    res = hiacsp.shrink(args.T)
                    if args.visualization:
                        plotCluster(res, hiacsp.label, 'LD-shrink', args, [])
                    total_time = time.time() - start
                    tbar.set_description('Dataset:{}'.format(args.data_name))
                    data_copy = hiacsp.res_data.copy()
                    # kmeans
                    if args.verbose:
                        print(prompt + " executing kmeans clustering")
                    for ss in range(5):
                        kmeans_res = KMeans(n_clusters=clusters, n_init='auto').fit_predict(data_copy)
                        para_map = {'K': args.K,
                                    'T': args.T,
                                    'D': i}
                        calculate_metric(hiacsp.label, kmeans_res, args, total_time, algo_name, 'kmeans', para_map)

                    # agg
                    if args.verbose:
                        print(prompt + " executing agg clustering")
                    agg_res = AgglomerativeClustering(n_clusters=clusters).fit_predict(data_copy)
                    para_map = {'K': args.K,
                                'T': args.T,
                                'D': i}
                    calculate_metric(hiacsp.label, agg_res, args, total_time, algo_name, 'agg', para_map)

                    # dpc
                    if args.verbose:
                        print(prompt + " executing dpc clustering")
                    dpc = DPC(n_clusters=clusters, center='auto')
                    dpc_res = dpc.fit_predict(data_copy)
                    para_map = {'K': args.K,
                                'T': args.T,
                                'D': i}
                    calculate_metric(hiacsp.label, dpc_res, args, total_time, algo_name, 'dpc', para_map)
                    tbar.update(1)
