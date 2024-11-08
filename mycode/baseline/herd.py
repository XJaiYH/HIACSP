import os
import time
import sys
sys.path.append('../../')
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, OPTICS, SpectralClustering, DBSCAN, MeanShift, \
    AffinityPropagation
from tqdm import tqdm
import argparse
import numpy as np
import scipy.spatial.distance as dis

from mycode.util.DPC import DPC
from mycode.util.dataloader import load_dataset
from mycode.util.evaluate import calculate_metric
import warnings
warnings.filterwarnings('ignore')

def force(threshold,data,v):
    distance = dis.pdist(data)
    distance_matrix = dis.squareform(distance)
    for i in range(data.shape[0]):
        a = np.zeros([data.shape[1]])
        total = 0
        for j in range(data.shape[0]):
            if i==j:
                continue
            if distance_matrix[i,j]<threshold:
                a = a + (data[j,:]-data[i,:])
                total = total + 1
        if total != 0:
            v[i,:] = v[i,:] + a/total
        if np.linalg.norm(v[i,:])>(threshold/2):
            v[i, :] = (threshold/2)/np.linalg.norm(v[i,:])*v[i, :]

    data = data + v
    return data, v

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--data_dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='wine.txt', type=str)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--with_label', type=bool, default=True)
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    args = parser.parse_args()
    data, label = load_dataset(args)
    if os.path.exists(args.save_dir) is False:
        os.mkdir(args.save_dir)
    if os.path.exists(args.save_dir + args.data_name) is False:
        os.mkdir(args.save_dir + args.data_name)
    prompt = "【prompt information】 "

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

    with tqdm(total=10) as tbar:
        for i in range(10):
            if args.verbose:
                print(prompt + " in the " + str(i+1) + "-th iteration.")
            threshold = (i + 1) * 0.1
            v = np.zeros(data.shape)
            data_copy = data.copy()

            total_time = 0
            for j in range(100):
                tbar.set_description('Dataset:{}'.format(args.data_name))
                start = time.time()
                data_copy, v = force(threshold, data_copy, v)
                total_time += time.time() - start

            # kmeans
            if args.verbose:
                print(prompt + " executing kmeans clustering")
            for ss in range(5):
                kmeans_res = KMeans(n_clusters=cluster_num, n_init='auto').fit_predict(data_copy)
                para_map = {
                    'threshold': threshold
                }
                calculate_metric(label, kmeans_res, args, total_time, 'Herd-MAX', 'kmeans', para_map)

            # agg
            if args.verbose:
                print(prompt + " executing agg clustering")
            agg_res = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(data_copy)
            para_map = {
                'threshold': threshold
            }
            calculate_metric(label, agg_res, args, total_time, 'Herd-MAX', 'agg', para_map)

            if args.verbose:
                print(prompt + " executing dpc clustering")
            dpc = DPC(n_clusters=cluster_num, center='auto')
            dpc_res = dpc.fit_predict(data_copy)
            para_map = {
                'threshold': threshold
            }
            calculate_metric(label, dpc_res, args, total_time, 'Herd-MAX', 'dpc', para_map)

            tbar.update(1)
