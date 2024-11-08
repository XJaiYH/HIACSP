# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: Xianjun Z
"""

import sys
sys.path.append('../../')
import argparse
import os.path
import time
import json
import numpy as np
import pandas as pd
import scipy.spatial.distance as dis
from sklearn.cluster import KMeans, Birch, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn import decomposition as dec
from sklearn.neighbors import KDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import warnings

from mycode.util import dataloader
from mycode.util.DPC import DPC
from mycode.util.evaluate import calculate_metric
warnings.filterwarnings('ignore')

global num
num = 1

global border
border = []

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

def plotCluster(data: np.ndarray, labels: np.ndarray, title: str, args):
    fig, ax = plt.subplots()
    label_set = set(labels)
    color = [
        '#606470',
        '#db6400',
        "#900C3F",  # 紫红色
        "#006400",  # 深绿色
        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
        "#A9A9A9",  # 暗灰色
        "#556B2F",  # 暗绿色
        "#9370DB",  # 中紫色
        "#8B7355",  # 赭色
        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#008B8B",  # 暗藏青色
        "#BDB76B",  # 黄褐色
        "#654321",  # 深棕色
        "#9400D3",  # 暗紫色
        "#008080",  # 暗青色
        "#CD5C5C"  # 褐红色
    ]
    lineform = ['o']
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        fontSize = 30
        colorNum = i % len(color)
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(title, fontsize=20)
    save_path = args.save_dir + args.data_name
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    save_path += "/" + title + ".png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def TGP(data, k, threshold, title, args):
    '''

    Parameters
    ----------
    data
    k
    photo_path: the address to save decision-graph
    threshold

    Returns:the edge weight matrix that record the edge weight of each object i and its k-nearest-neighbors
    -------

    '''
    global num
    pointNum = data.shape[0]
    probability = np.zeros([int(pointNum / 10), 2])
    tree = KDTree(data)
    distance_sort, disIndex = tree.query(data, max(k + 2, round(pointNum * 0.015) + 1))
    max_dis = np.max(distance_sort)
    distance_sort = -(distance_sort - max_dis)
    # distance = dis.pdist(data)
    # distance_matrix = dis.squareform(distance) - np.max(distance)
    # distance_sort = -np.sort(distance_matrix, axis=1)

    pointWeight = np.mean(distance_sort[:, 1:k + 1], axis=1)  # get point weight to replace edge weight
    dc = (max(pointWeight) - min(pointWeight)) * 10 / pointNum# get the length of each interval
    for i in range(probability.shape[0]):  # for each interval
        location = min(pointWeight) + dc * i  # get the start position of each interval
        probability[i, 0] = location  # the start position is recorded at probability[:, 0]
    for i in range(pointNum):  # for each object
        j = (int)((pointWeight[i] - min(pointWeight)) / dc)  # for each object, calculate the interval to which it belongs
        if j < int(pointNum / 10):
            probability[j, 1] += 1
    probability[:, 1] = probability[:, 1] / pointNum

    # fig = plt.figure()
    # num += 1
    # plt.plot(probability[:, 0], probability[:, 1])
    # plt.title('Decision graph', fontstyle='italic', size=20)
    # plt.xlabel('wight', fontsize=20)
    # plt.ylabel('probability', fontsize=20)
    # # points = plt.ginput(n=-1, timeout=-1)
    # save_path = args.save_dir + args.data_name
    # if os.path.isdir(save_path) == False:
    #     os.mkdir(save_path)
    # save_path += "/2-" + title + ".png"
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # for point in points:
    #     pos = point[0]
    #     border.append(pos)
    return distance_sort

def prune(data, knn, threshold, distanceTGP):
    '''

    Parameters
    ----------
    data:
    knn:the number of neighbor
    threshold:to clip invalid-edge
    distanceTGP:the weight matrix for each object, we only need distanceTGP[:,:k+1], i.e. the k-nearest-neighbors

    Returns: the index matrix which records the valid-neighbors index of object i
    -------

    '''
    pointNum = data.shape[0]
    tree = KDTree(data)
    distance_sort, disIndex = tree.query(data, max(knn + 2, round(pointNum * 0.015) + 1))
    # distance = dis.pdist(data)
    # distance_matrix = dis.squareform(distance)
    # disIndex = np.argsort(distance_matrix, axis=1)
    # distance_sort = np.sort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(data.shape[1] * 0.015)])
    density = tree.query_radius(data, area, count_only=True)

    # density = np.zeros(pointNum)
    # for i in range(pointNum):
    #     num = 1
    #     while (distance_sort[i, num] < area):
    #         num += 1
    #     density[i] = num
    densityThreshold = np.mean(density)  # we didn't move objects that have high density

    for i in range(pointNum):
        if density[i] < densityThreshold:
            for j in range(knn + 1):  # for k nearest neighbours
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
                    # to clip invalid-neighbors
                    if distanceTGP[i, j] < threshold:
                        disIndex[i][j] = -1
    return disIndex

def shrink(data, knn, T, disIndex):
    '''

    Parameters
    ----------
    data
    knn
    T:
    disIndex:i.e. the index matrix which records the valid-neighbors index of each object i
            for object i, if j is invalid-neighbor of i, neighbor_index[i][j] = -1,
            else neighbor_index[i][j] is the index of object j
    Returns:dataset after ameliorating
    -------

    '''
    bata = data.copy()
    pointNum = data.shape[0]
    tree = KDTree(data)
    distance_sort, idx_matrix = tree.query(data, max(knn + 2, round(pointNum * 0.015) + 1))

    # distance = dis.pdist(data)
    # distance_matrix = dis.squareform(distance)
    # distance_sort = np.sort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(pointNum * 0.015)])
    density = tree.query_radius(bata, area, count_only=True)

    # calculate density of each object
    # density = np.zeros(pointNum)
    # for i in range(pointNum):
    #     num = 1
    #     while (distance_sort[i, num] < area):
    #         num += 1
    #     density[i] = num

    densityThreshold = np.mean(density)
    G = np.mean(distance_sort[:, 1])  # Gravitational constant for each time-segment

    # move the objects to ameliorate the dataset
    for i in range(pointNum):  # for each object
        if density[i] < densityThreshold:  # we didn't move objects that have high density
            displacement = np.zeros(data.shape[1], dtype=np.float32)

            # calculate graduation pull of all valid neighbours on current object i, and then calculate the displacement of current object i.
            for j in range(knn + 1):  # for k nearest neighbours
                if (data[disIndex[i][j]] == data[i]).all():  # if itself
                    continue
                else:
                    if disIndex[i][j] != -1:
                        ff = (data[disIndex[i][j]] - data[i])
                        fff = (distance_sort[i, 1] / (
                                    np.sum(np.square(data[i]-data[disIndex[i, j]]))))
                        displacement += G * ff * fff
            bata[i] = data[i] + displacement * T  # object after moving
    return bata

if __name__ == "__main__":
    ######################initialization and parameter config#################################
    global photoPath
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--data_dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--th_dir', default='../../data/', type=str, help='the director of threshold files')
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--data_name', default='aggregation.txt', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1,'
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--data_id', type=int, default=850)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--load_data_local', type=bool, default=True)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    parser.add_argument('--with_label', default=True, type=bool)
    parser.add_argument('--k', default=20, type=int, help="the number of nearest neighbors, parameter k in HIAC")
    parser.add_argument('--T', default=0.5, type=float, help='parameter T in HIAC')
    parser.add_argument('--d', default=40, type=int, help='the d in paper HIAC')
    parser.add_argument('--threshold', default='0.728255891455162,0.7034055327224511,0.7111362859854794,0.6909900528514239,0.6914504952122373,0.6779111710744872,0.675184790533747,0.6647877096372503,0.660685028803071,0.6546931568683835'
                                               , type=str, help='the weight threshold to clip invalid-neighbors')
    args = parser.parse_args()
    prompt = "【prompt information】 "
    data, label = load_data(args)
    #print(data.shape)
    if os.path.isdir(args.save_dir) is False:
        os.makedirs(args.save_dir)
    pca = dec.PCA(n_components=2)# High-dimensional data are displayed using PCA dimensionality reduction methods

    cluster_num = len(set(label))# the number of clusters

    ########################normalization###################################
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)

    ########################use PCA to reduce the dimension of the dataset, and visualization###################################
    if args.show_fig:
        if data.shape[0] > 2:# for high dimension dataset
            dataPCA = pca.fit_transform(data)
            plotCluster(dataPCA, label, "original", args)
        else:# for dataset has two dimensions
            plotCluster(data, label, "original", args)

    # k_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    k_set = [24]
    T_set = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # T_set = [2]

    if args.ratio != 0:
        path = args.data_dir + args.data_name + '/' + args.data_name + '-' + str(args.ratio) + '-threshold.json'
    else:
        path = args.data_dir + args.data_name + '/' + args.data_name + '-threshold.json'
    with open(path, 'r') as f:
        content = f.read()
        thresholds = json.loads(content)
    total_th = 0
    for key, item in thresholds.items():
        total_th += len(item)
    borders = { }


    ######################call HIAC##############################
    with tqdm(total=len(T_set) * total_th * 10) as tbar:
        tbar.set_description("【prompt information】 ")
        for ss, k in enumerate(k_set):
            border = []
            args.k = k
            th_set = thresholds[str(k)]
            print(th_set)
            for th in th_set:
                start = time.time()
                distanceTGP = TGP(data.copy(), args.k, th, "decision_" + str(args.k) + "_" + str(th),
                                  args)  # we can determine the threshold，and return the weight matrix
                # th = border[0]
                neighbor_index = prune(data.copy(), args.k, th,
                                       distanceTGP) # clip invalid-neighbors based on the weight threshold and the decision-graph,
                                                    # and then return the index matrix which records the valid-neighbors index of object i
                                                    # for object i, if j is invalid-neighbor of i, neighbor_index[i][j] = -1,
                                                    # else neighbor_index[i][j] is the index of object j
                                                    # its necessary for you to know that we only need K-nearest-neighbor of each object,
                                                    # so,
                add_time = time.time() - start
                # print(border)
                # borders[k] = border
                # continue
                for T in T_set:
                    args.T = T
                    total_time = add_time
                    tbar.set_postfix({'th': th, 'T': T, 'k': k})
                    data_copy = data.copy()

                    for i in range(args.d): # ameliorated the dataset by d time-segments
                        start = time.time()
                        bata = shrink(data_copy, args.k, args.T, neighbor_index)
                        data_copy = bata
                        total_time += time.time() - start
                        # print('time: ', total_time)
                        # plotCluster(data_copy, label, "tmp-shrink", args)
                        if (i+1) % 4 == 0:
                            # plotCluster(data_copy, label, "tmp-shrink", args)
                            # np.savetxt(args.save_dir + args.data_name + '/ameliorated-' + str(i) + '.txt', data_copy)
                            # kmeans
                            if args.verbose:
                                print(prompt + " executing kmeans clustering")
                            kmeans_res = KMeans(n_clusters=cluster_num, n_init='auto').fit_predict(data_copy)
                            para_map = {'K': k,
                                        'T': T,
                                        'D': i,
                                        'th': th}
                            calculate_metric(label, kmeans_res, args, total_time, 'HIAC', 'kmeans', para_map)
                            #
                            # # agg
                            if args.verbose:
                                print(prompt + " executing agg clustering")
                            agg_res = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(data_copy)
                            para_map = {'K': k,
                                        'T': T,
                                        'D': i,
                                        'th': th}
                            calculate_metric(label, agg_res, args, total_time, 'HIAC', 'agg', para_map)

                            if args.verbose:
                                print(prompt + " executing dpc clustering")
                            dpc = DPC(n_clusters=cluster_num, center='auto')
                            dpc_res = dpc.fit_predict(data_copy)
                            para_map = {
                                'K': k,
                                'T': T,
                                'D': i,
                                'th': th
                            }
                            calculate_metric(label, dpc_res, args, total_time, 'HIAC', 'dpc', para_map)
                        tbar.update(1)

        # path = args.data_dir + args.data_name + '/' + args.data_name + '-threshold.json'
        # with open(path, 'w') as f:
        #     json.dump(borders, f)