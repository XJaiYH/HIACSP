import argparse
import os
import time
import sys
sys.path.append('../../')
import numpy as np
import math
import scipy.spatial.distance as dis
import sklearn.cluster as sc
from sklearn import metrics
from sklearn.cluster import SpectralClustering, DBSCAN, Birch, AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine    #改改改改改改
from tqdm import tqdm

from mycode.util.DPC import DPC
from mycode.util.dataloader import load_dataset
import warnings

from mycode.util.evaluate import calculate_metric

warnings.filterwarnings('ignore')


def shrink_SCBA(data, k):
    #归一化
    # for i in range(data.shape[1]):
    #     max_ = max(data[:, i])
    #     min_ = min(data[:, i])
    #     for j in range(data.shape[0]):
    #         data[j, i] = (data[j, i] - min_) / (max_ - min_)
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)

    #识别数据所在的cell
    grid_location = np.zeros(data.shape)
    grid_ID = np.zeros(data.shape[0])
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            grid_location[i,j]=int(data[i,j]*k)

    #识别非空的cell并计算其内的元素个数
    cell = []
    cell_sum = []
    cell_ss = 0
    for i in range(grid_location.shape[0]):
        if grid_location[i,:].tolist() not in cell:
            cell.append(grid_location[i,:].tolist())
            cell_sum.append(1)
            grid_ID[i] = cell_ss
            cell_ss = cell_ss+1
        else:
            index = cell.index(grid_location[i,:].tolist())
            cell_sum[index] = cell_sum[index]+1
            grid_ID[i] = index
    cell = np.array(cell)
    cell_sum = np.array(cell_sum)
    #print(cell.shape)

    #识别dense cell 还是 sparse cell
    Tdn1 = 1/3*(np.mean(cell_sum))
    cell_judge_dense = np.ones(cell_sum.shape)
    for i in range(cell.shape[0]):
        if cell_sum[i]<Tdn1:
            cell_judge_dense[i]=-1
    #print(cell_judge_dense)

    #计算cell的中心
    cell_centroid = np.zeros(cell.shape)
    for i in range(cell_centroid.shape[0]):
        cell_centroid_ss = 0
        for j in range(data.shape[0]):
            if grid_ID[j] == i:
                cell_centroid[i,:] = cell_centroid[i,:] + data[j,:]
                cell_centroid_ss = cell_centroid_ss +1.
        cell_centroid[i,:] = cell_centroid[i,:]/cell_centroid_ss
    #print(cell_centroid)

    #计算cell邻居的中心(都为零是sparse cell 或表示没有dense邻居的cell)
    cell_neighbor_centroid = np.zeros(cell.shape)
    cell_neighbor_sum = np.zeros(cell_sum.shape)
    for i in range(cell.shape[0]):
        if cell_judge_dense[i]==-1:
            continue
        for j in range(cell.shape[0]):
            if i==j:
                continue
            flag = True
            for z in range(cell.shape[1]):
                if abs(cell[i,z]-cell[j,z])>1:
                    flag=False
            if flag and cell_judge_dense[j]==1:
                cell_neighbor_centroid[i,:] = cell_neighbor_centroid[i,:] + cell_centroid[j,:]*cell_sum[j]
                cell_neighbor_sum[i] = cell_neighbor_sum[i] + cell_sum[j] #sparse cell 和 没有cell邻居的cell的cell_neighbor_centroid_sum都为零
        if cell_neighbor_sum[i]!=0:
            cell_neighbor_centroid[i, :] = cell_neighbor_centroid[i, :] / cell_neighbor_sum[i]
    #print(cell_neighbor_centroid)

    #数据移动
    Tmv = 0.5*math.sqrt(data.shape[1])
    for i in range(cell.shape[0]):
        if cell_judge_dense[i]==-1:
            continue
        if np.linalg.norm(cell_neighbor_centroid[i,:]-cell_centroid[i,:])>=Tmv/k and cell_neighbor_sum[i] >cell_sum[i]:
            for j in range(data.shape[0]):
                if grid_ID[j]==i:
                    data[j,:]=data[j,:]+cell_neighbor_centroid[i,:]-cell_centroid[i,:]

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--data_dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='wine.txt', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--with_label', type=bool, default=True)
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool) # raisin 850
    args = parser.parse_args()
    data, label = load_dataset(args)
    prompt = "【prompt information】 "
    if os.path.exists(args.save_dir) is False:
        os.mkdir(args.save_dir)
    if os.path.exists(args.save_dir + args.data_name) is False:
        os.mkdir(args.save_dir + args.data_name)

    cluster_num = len(set(label.tolist()))
    if args.verbose:
        print(prompt + " for dataset" + args.data_name)
    with tqdm(total=20) as tbar:
        Tamv = 0.024
        for zz in range(20):
            data_copy = data.copy()
            stop_flag = np.zeros(2)
            if zz == 0:
                k = 5
            else:
                k = zz * 10 + 5

            total_time = 0
            start = time.time()
            for i in range(10):
                stop_flag[0] = stop_flag[1]
                bata = shrink_SCBA(data_copy, k)
                stop_the = np.mean(np.linalg.norm((bata - data_copy), axis=1))
                data_copy = bata
                if stop_the <= Tamv:
                    stop_flag[1] = 1
                else:
                    stop_flag[1] = 0
                if stop_flag[0] == 1 and stop_flag[1] == 1:
                    break

            total_time = time.time() - start
            # kmeans
            if args.verbose:
                print(prompt + " executing kmeans clustering")
            for ss in range(5):
                kmeans_res = KMeans(n_clusters=cluster_num, n_init='auto').fit_predict(data_copy)
                para_map = {
                    'K': k,
                    'D': i
                }
                calculate_metric(label, kmeans_res, args, total_time, 'SBCA-MAX', 'kmeans', para_map)

            # agg
            if args.verbose:
                print(prompt + " executing agg clustering")
            agg_res = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(data_copy)
            para_map = {
                'K': k,
                'D': i
            }
            calculate_metric(label, agg_res, args, total_time, 'SBCA-MAX', 'agg', para_map)

            # dpc
            if args.verbose:
                print(prompt + " executing dpc clustering")
            dpc = DPC(n_clusters=cluster_num, center='auto')
            dpc_res = dpc.fit_predict(data_copy)
            para_map = {
                'K': k,
                'D': i
            }
            calculate_metric(label, dpc_res, args, total_time, 'SBCA-MAX', 'dpc', para_map)

            tbar.update(1)
            # data_tmp = np.concatenate([data_copy, label.reshape(-1, 1)], axis=1)
            # np.savetxt(args.save_dir + args.data_name + '/' + 'ameliorated' + str(zz) + '.txt', data_tmp)


