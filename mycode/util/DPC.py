""" Clustering by fast search and find of density peaks."""

# Author: Zhong Tiansheng <bitzhong@outlook.com>

import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dis
from matplotlib.widgets import RectangleSelector

from mycode.util.visualization import plotCluster


class DPC:
    """
    DPC clustering.

    Parameters
    ----------

    d_c_ratio: float, default=0.02
        截断距离内平均样本数占总样本数的百分比.

    n_clusters: int, default=8
        当中心点选取模式为"auto"时才有效，程序会自动选取密度和delta乘积值最高的n_clusters个点作为中心点.

    center: string, default='manual'
        中心点选取模式，设置为"manual"时，从决策图中框选；设置为"auto"时，程序根据密度和
        delta乘积值自动选择n_clusters个中心点；设置为"given"时，中心点由用户在fit_predict()
        的参数中指定.

    density: string, default='cutoff'
        密度计算方法，有"cutoff""gauss"和"precomputed"三个可选选项，当设置为"precomputed"时密度由用户在fit_predict()
        的参数中给定.

    distance: string, default='normal'
        距离计算方法，有"normal""island"和"precomputed"三个可选选项，当设置为"precomputed"时距离矩阵由用户在fit_predict()
        的参数中给定.

    Attributes
    ----------
    labels_: ndarray of shape (n_samples)
        labels of each point.

    delta_: ndarray of shape (n_samples)
        delta.

    parent_: list
        the index of the nearest point among those having a larger density than i.

    dist_matrix_: ndarray of shape (n_samples, n_samples)
        distance matrix.

    dist_sort_: ndarray of shape (n_samples, n_samples)
        the distance matrix sorted by row.

    density_: ndarray of shape (n_samples)
        denisty.

    density_index_: ndarray of shape (n_samples)
        point indexes sorted by density descend.

    centers_: list
        index of cluster centers.
    """
    labels_ = None
    delta_ = None
    parent_ = None
    dist_matrix_ = None
    dist_sort_ = None
    density_ = None
    density_index_ = None
    centers_ = None

    def __init__(self, d_c_ratio=0.02, n_clusters=8, center='manual', density='cutoff', distance='normal'):
        self.d_c_ratio = d_c_ratio
        self.n_clusters = n_clusters
        self.center = center
        self.density = density
        self.distance = distance

    def get_centers(self):
        # 返回centers,是一个list列表
        return self.centers_

    def line_select_callback(self, eclick, erelease):  # 框选的回调
        """--
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        centers = []
        # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        # print(f" The buttons you used were: {eclick.button} {erelease.button}")
        for index in reversed(self.density_index_):
            if self.density_[index] >= x1:
                if self.delta_[index] >= y1:
                    centers.append(index)
                    # print(density[index])
                    # print(delta[index])
            else:
                break
        self.centers_ = centers.copy()
        #print(f"{centers}:{len(centers)}")
        return

    def plot_decision(self):
        fig1, ax1 = plt.subplots()
        ax1.scatter(self.density_, self.delta_)
        ax1.set_title(
            "(DPC)Click and drag to rectangle-select cluster centers.\n")
        # ax1.scatter(density[clustering_center], delta[clustering_center], s=10, c='red')
        RS = RectangleSelector(ax1, self.line_select_callback,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # disable middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        plt.show()

    def fit_predict(self, X, density=None, centers=None):
        """

        :param X: ndarray of shape (n_samples, n_features) or ndarray of shape (n_samples, n_samples)
        数据点或距离矩阵
        :param density: ndarray of shape (n_samples) 密度数组
        :param centers: list 中心点在样本点中的序号
        :return: labels: ndarray of shape (n_samples) 标签
        """
        n_samples = X.shape[0]
        if self.distance == 'precomputed':
            self.dist_matrix_ = X
            self.dist_sort_ = np.sort(X, axis=1)
        else:
            self.get_distance(X)

        if self.density == 'precomputed':
            self.density_ = density
            self.density_index_ = np.argsort(density)
        else:
            self.get_density(n_samples)

        self.get_delta(n_samples)

        if self.center == 'given':
            self.centers_ = centers
        elif self.center == 'manual':
            self.plot_decision()
        elif self.center == 'auto':
            self.select_centers(n_samples)
        self.get_labels(n_samples)
        return self.labels_

    def select_centers(self, n_samples):
        gamma = np.multiply(self.density_, self.delta_)
        clustering_center = np.argsort(gamma)[n_samples - self.n_clusters:]
        self.centers_ = clustering_center.tolist()
        #print(f"{self.centers_}:{len(self.centers_)}")
        return

    def get_distance(self, X):
        if self.distance == 'normal':
            # 求距离（常规）
            distance = dis.pdist(X)
            self.dist_matrix_ = dis.squareform(distance)
            self.dist_sort_ = np.sort(self.dist_matrix_, axis=1)
        elif self.distance == 'island':
            # 求距离(孤岛距离)
            dim = X.shape[0]
            distance = dis.pdist(X)
            b_matrix = dis.squareform(distance)
            b_sort = np.sort(b_matrix, axis=1)
            # b_index = np.argsort(b_matrix, axis=1)
            distance_matrix = np.zeros([dim, dim])
            for i in range(dim):
                for j in range(i + 1, dim):
                    distance_matrix[i, j] = max(
                        [b_sort[i, round(dim * 0.015)], b_sort[j, round(dim * 0.015)], b_matrix[i, j]])
                    distance_matrix[j, i] = distance_matrix[i, j]
            distance_sort = np.sort(distance_matrix, axis=1)
            self.dist_matrix_ = distance_matrix.copy()
            self.dist_sort_ = distance_sort.copy()

    def get_density(self, row_num):
        density = np.zeros(row_num)
        area = np.mean(self.dist_sort_[:, round(row_num * self.d_c_ratio)])
        if self.density == 'cutoff':
            # 求密度(剪切密度)
            for i in range(row_num):
                s = 1
                while self.dist_sort_[i, s] < area:
                    s = s + 1
                density[i] = s
        elif self.density == 'gauss':
            # 求密度(高斯密度)
            for i in range(row_num - 1):
                for j in range(i + 1, row_num):
                    density[i] = density[i] + math.exp(- (self.dist_matrix_[i][j] * self.dist_matrix_[i][j]) / (area * area))
                    density[j] = density[j] + math.exp(- (self.dist_matrix_[i][j] * self.dist_matrix_[i][j]) / (area * area))
            # density_sort = np.sort(density)
        self.density_ = (density - np.amin(density))/(np.amax(density) - np.amin(density))
        density_index = np.argsort(density)  # 对密度排序，得到排序序号
        self.density_index_ = density_index.copy()

    def get_delta(self, row_num):
        parent = [0 for i in range(row_num)]
        # max_dis = np.amax(pdist_matrix)
        max_delta = 0
        delta = np.zeros(row_num)
        density_index = self.density_index_[::-1]
        for i in range(row_num):
            delta[density_index[i]] = float('inf')
            for j in range(i):
                dij = self.dist_matrix_[density_index[i]][density_index[j]]
                if dij < delta[density_index[i]]:
                    delta[density_index[i]] = dij
                    parent[density_index[i]] = density_index[j]
                    if dij > max_delta:
                        max_delta = dij
        delta[density_index[0]] = max_delta * 1.1
        self.delta_ = (delta - np.amin(delta))/(np.amax(delta) - np.amin(delta))
        self.parent_ = parent.copy()

    # def get_centers(self, density, delta, row_num):
    #     den_max = np.amax(density)
    #     den_min = np.amin(density)
    #     den_range = den_max - den_min
    #     del_max = np.amax(delta)
    #     del_min = np.amin(delta)
    #     del_range = del_max - del_min
    #     gamma = np.zeros(row_num)
    #     for i in range(row_num):
    #         density[i] = (density[i] - den_min) / den_range
    #         delta[i] = (delta[i] - del_min) / del_range
    #         gamma[i] = density[i] * delta[i]
    #     clustering_center = np.argsort(gamma)[row_num - self.n_clusters:]
    #     return clustering_center

    def get_labels(self, row_num):
        # dim = len(parent)
        label = np.zeros(row_num, dtype=np.int8)
        density_index = self.density_index_[::-1]
        i = 1
        for c in self.centers_:
            label[c] = i
            i = i + 1

        for index in density_index:
            if label[index] == 0:
                label[index] = label[self.parent_[index]]

        self.labels_ = label.copy()