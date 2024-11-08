# Capable of handling TXT, CSV, and MAT file types,
# You can also download the dataset from uci by function "load_dataset_from_UCI"
import numpy as np
import pandas as pd
import scipy.io as si
from scipy.io import arff
import collections
import os

def load_dataset(args, ty='csv'):
    if args.data_name.__contains__('.'):
        name, ty = args.data_name.split(".")
        args.data_name = name
    if ty == 'txt':
        if args.ratio == 0:
            data = np.loadtxt(args.data_dir + args.data_name + "/" + args.data_name + '.txt', dtype=np.float64)
        else:
            data = np.loadtxt(args.data_dir + args.data_name + "/" + args.data_name + '-' + str(args.ratio) + '.txt', dtype=np.float64)
    elif ty == 'csv':
        if args.ratio == 0:
            data = pd.read_csv(args.data_dir + args.data_name + "/" + args.data_name + '.csv').values
        else:
            data = pd.read_csv(args.data_dir + args.data_name + "/" + args.data_name + '-' + str(args.ratio) + '.csv', dtype=np.float64)
    elif ty == 'mat':
        data = si.loadmat(args.data_dir + args.data_name + "/" + args.data_name + '.mat')
        X = data['X']
        y = data['y']
        data = np.concatenate([X, y], axis=1)
    if args.with_label is True:
        label = data[:, -1].astype(int)
        data = data[:, :-1]
    else:
        label = np.loadtxt(args.data_dir + args.data_name + "/" + args.data_name + '-label.txt').astype(int).reshape(-1, )
        # label = np.zeros((data.shape[0], ))
    return data, label

def load_dataset_from_UCI(id: int):
    from ucimlrepo import fetch_ucirepo
    # fetch dataset
    zoo = fetch_ucirepo(id=id)
    # data (as pandas dataframes)
    X = zoo.data.features
    y = zoo.data.targets
    # metadata
    print(X.head())
    return X.values, y.values.reshape(-1, )

def load_data(args):
    ty = args.data_name.split('.')[-1]
    if args.load_data_local:
        read_data, label = load_dataset(args, ty)
    else:
        read_data, label = load_dataset_from_UCI(args.data_id)
    for j in range(read_data.shape[1]):
        max_ = max(read_data[:, j])
        min_ = min(read_data[:, j])
        if max_ == min_:
            continue
        for i in range(read_data.shape[0]):
            read_data[i][j] = (read_data[i][j] - min_) / (max_ - min_)
    return read_data, label
