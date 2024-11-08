import argparse

import numpy as np
from sklearn.neighbors import KDTree

from mycode.util.dataloader import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../../data/')
parser.add_argument('--save_dir', type=str, default='../../result/')
parser.add_argument('--data_name', type=str, default='gamma.csv')
parser.add_argument('--load_data_local', type=bool, default=True)
parser.add_argument('--data_id', type=int, default=257)
parser.add_argument('--with_label', type=bool, default=True)
parser.add_argument('--visualization', type=bool, default=False)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--ratio', type=int, default=0)
parser.add_argument('--verbose', type=bool, default=False)
prompt = '【System Information: 】'
args = parser.parse_args()

data_set = ['waveform.csv', 'raisin.csv', 'imageSegmentation.csv', 'breast_cancer.csv', 'breastw.mat', 'forest.csv',
            'iris.txt', 'banknote.txt', 'HAPT.csv', 'crowdsourced.csv', 'seeds.txt', 'wine.txt', 'skewed.txt',
            'aggregation.txt', 'D31.txt', 'overlap1.csv', 'overlap2.csv']
for data_name in data_set:
    args.data_name = data_name
    data, label = load_data(args)
    # print(data_name + '\'s shape: {}'.format(data.shape[0]))
    tree = KDTree(data)
    dis_sort, nn_idx = tree.query(data, max(11, int(0.01 * data.shape[0])))
    is_overlap = np.zeros((data.shape[0], ), dtype=int)
    for i in range(data.shape[0]):
        if len(set(label[nn_idx[i]])) >= 2:
            # print(label[nn_idx[i]])
            is_overlap[i] = 1
    print('dataset: {}\'s overlap rate: {}, and shape: {},{}, and clusters: {}'
          .format(args.data_name, np.sum(is_overlap) / data.shape[0], data.shape[0], data.shape[1], len(set(label))))


data_set = ['R15.txt', 'two.txt', 'gauss17.txt', 'a1.txt']
ratios = ['10', '20', '30', '40', '50']
for data_name in data_set:
    for ratio in ratios:
        args.data_name = data_name
        args.ratio = ratio
        data, label = load_data(args)
        # print(data_name + '\'s shape: {}'.format(data.shape[0]))
        tree = KDTree(data)
        dis_sort, nn_idx = tree.query(data, max(11, int(0.01 * data.shape[0])))
        is_overlap = np.zeros((data.shape[0], ), dtype=int)
        for i in range(data.shape[0]):
            if len(set(label[nn_idx[i]])) >= 2:
                # print(label[nn_idx[i]])
                is_overlap[i] = 1
        print('dataset: {}\'s overlap rate: {}, and shape: {},{}, and clusters: {}'
              .format(args.data_name, np.sum(is_overlap) / data.shape[0], data.shape[0], data.shape[1], len(set(label))))