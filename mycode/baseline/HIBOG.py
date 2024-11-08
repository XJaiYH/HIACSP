# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: 佡儁
"""
import sys
sys.path.append('../../')
import argparse
import os.path
import time
from sklearn.metrics import adjusted_mutual_info_score as AMI
import numpy as np
import scipy.spatial.distance as dis
from sklearn.cluster import Birch, SpectralClustering, DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import KDTree
from tqdm import tqdm

from mycode.util import dataloader
from mycode.util.DPC import DPC
from mycode.util.dataloader import load_dataset
from mycode.util.evaluate import calculate_metric
import warnings
warnings.filterwarnings('ignore')


def shrink(data, num, step_length):
    bata = data.copy()
    dim = data.shape[0]
    tree = KDTree(data)
    distance_sort, distance_index = tree.query(data, max(num+2, round(dim*0.015)+1))
    # distance = dis.pdist(data)
    # distance_matrix = dis.squareform(distance)
    # distance_sort = np.sort(distance_matrix, axis=1)
    # distance_index = np.argsort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(dim*0.015)])
    density = np.zeros(dim)
    count = tree.query_radius(data, area, count_only=True)
    for i in range(dim):
        # listss = 1
        # while (distance_sort[i,listss]<area):
        #     listss = listss + 1
        density[i] = count[i]
        # density[i] = listss
    density_mean = np.mean(density)
    density_yuzhi = density_mean

    buchang = np.mean(distance_sort[:, 1])
    for i in range(dim):
        if density[i] >= density_yuzhi:
            list = []
        else:
            list = distance_index[i, 1:(num+1)]

        linshi = np.zeros(data.shape[1],dtype=np.float32)
        ss = 1
        for j in list:
            if (data[j] == data[i]).all():
                ss = ss + 1
            else:
                ff = ((data[j] - data[i]))
                fff = (distance_sort[i, 1] / (distance_sort[i, ss]*distance_sort[i, ss]))
                linshi = linshi + ff * buchang * fff
                ss = ss + 1
        bata[i] = data[i]+linshi*step_length
    return bata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--data_dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='overlap2.csv', type=str)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--with_label', type=bool, default=True)
    parser.add_argument('--load_data_local', type=bool, default=True)
    parser.add_argument('--data_id', type=int, default=257)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    args = parser.parse_args()
    ty = args.data_name.split('.')[-1]
    if args.load_data_local:
        data, label = dataloader.load_dataset(args, ty)
    else:
        data, label = dataloader.load_dataset_from_UCI(args.data_id)
    # label = np.zeros((data.shape[0], ), dtype=int)
    prompt = "【prompt information】 "
    if os.path.isdir(args.save_dir + '/' + args.data_name) is False:
        os.mkdir(args.save_dir + '/' + args.data_name)

    ######### # normalization # ###############
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)

    cluster_num = len(set(label.tolist()))

    T_set = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]#bank会稍微大点
    # T_set = [0.5]
    K_set = [12, 14, 16, 18, 20, 22, 24]
    # K_set = [15]
    iterationTime = 10

    with tqdm(total=len(T_set) * len(K_set) * 10) as tbar:
        for k in K_set:
            for t in T_set:
                data_copy = data.copy()
                total_time = 0

                for j in range(iterationTime):
                    start = time.time()
                    data_copy = shrink(data_copy, k, t)
                    total_time += time.time() - start
                    tbar.set_description('Dataset:{}'.format(args.data_name))
                    # plotCluster(data_copy, label, 'test', args)
                    # data_tmp = np.concatenate([data_copy, label.reshape(-1, 1)], axis=1)
                    # kmeans
                    if args.verbose:
                        print(prompt + " executing kmeans clustering")
                    kmeans_res = KMeans(n_clusters=cluster_num, n_init='auto').fit_predict(data_copy)
                    para_map = {'K': k,
                                'T': t,
                                'D': j}
                    calculate_metric(label, kmeans_res, args, total_time, 'HIBOG', 'kmeans', para_map)
                    # agg
                    if args.verbose:
                        print(prompt + " executing agg clustering")
                    agg_res = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(data_copy)
                    para_map = {'K': k,
                                'T': t,
                                'D': j}
                    calculate_metric(label, agg_res, args, total_time, 'HIBOG', 'agg', para_map)
                    # dpc
                    if args.verbose:
                        print(prompt + " executing dpc clustering")
                    dpc = DPC(n_clusters=cluster_num, center='auto')
                    dpc_res = dpc.fit_predict(data_copy)
                    para_map = {'K': k,
                                'T': t,
                                'D': j}
                    calculate_metric(label, dpc_res, args, total_time, 'HIBOG', 'dpc', para_map)
                    tbar.update(1)
                # np.savetxt(args.save_dir + args.data_name + '/ameliorated.txt', data_tmp)


