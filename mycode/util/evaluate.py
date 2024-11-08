# evaluation metrics
import os

import numpy as np
import pandas as pd


def computePurity(labels_true, labels_pred):
  clusters = np.unique(labels_pred)
  labels_true = np.reshape(labels_true, (-1, ))
  labels_pred = np.reshape(labels_pred, (-1, ))
  count = []
  for c in clusters:
    idx = np.where(labels_pred == c)
    labels_tmp = labels_true[idx].reshape((-1, )).astype(int)
    count.append(np.bincount(labels_tmp).max())
  return np.sum(count) / labels_true.shape[0]


def calculate_metric(labels, pred, args, clustering_time, algorithm_name, clustering_name, para_map):
    # clustering by kmeans
    from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, \
    fowlkes_mallows_score as FMI, homogeneity_completeness_v_measure as HCV

    ari = ARI(labels, pred)
    nmi = AMI(labels, pred)
    fmi = FMI(labels, pred)
    purity = (computePurity(labels, pred))
    hcv = HCV(labels, pred)
    homogeneity = (hcv[0])
    completeness = (hcv[1])
    vm = (hcv[2])
    my_metrics = [[ari, nmi, fmi, purity, vm, homogeneity, completeness, clustering_time]]
    my_column = ['ari', 'nmi', 'fmi', 'purity', 'vm', 'homogeneity', 'completeness', 'time']
    key_list = []
    val_list = []
    for key, val in para_map.items():
        key_list.append(key)
        val_list.append(val)
    if len(key_list) > 0:
        my_column.extend(key_list)
        my_metrics[0].extend(val_list)
    # print(my_metrics)
    # print(my_column)
    my_metrics = pd.DataFrame(my_metrics, columns=my_column)
    save_path = args.save_dir + args.data_name
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    if args.ratio != 0:
        save_path = save_path + "/result-" + algorithm_name + "-" + clustering_name + '-' + str(args.ratio) + ".csv"
    else:
        save_path = save_path + "/result-" + algorithm_name + "-" + clustering_name + ".csv"
    if os.path.exists(save_path):
        my_metrics.to_csv(save_path, index=False, header=None, mode='a')
    else:
        my_metrics.to_csv(save_path, index=False)