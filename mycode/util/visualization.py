import os

import matplotlib.pyplot as plt
import numpy as np
import random

import pandas as pd

def plotCluster(data: np.ndarray, labels: np.ndarray, title: str, args, over_mean=None, nneigh=None):
    fig, ax = plt.subplots()
    label_set = set(labels.tolist())
    color = [
        "#900C3F",  # 紫红色
        "#556B2F",  # 暗绿色
        "#9370DB",  # 中紫色
        "#8B7355",  # 赭色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
        "#A9A9A9",  # 暗灰色
        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#008B8B",  # 暗藏青色
        "#006400",  # 深绿色
        "#BDB76B",  # 黄褐色
        "#4B0082",  # 靛青色
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
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[int(colorNum)], lineform[int(formNum)])
    if over_mean is not None:
        plt.scatter(data[over_mean, 0], data[over_mean, 1], fontSize, '#4B0082', lineform[int(formNum)])
    if nneigh is not None:
        for i in range(nneigh.shape[0]):
            tmp = data[nneigh[i]] - data[i]
            plt.arrow(data[i][0], data[i][1], tmp[0], tmp[1], shape='full')

    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.title(title, fontsize=20, family='Times New Roman')
    plt.xticks([])
    plt.yticks([])
    if args is not None:
        save_path = args.save_dir + args.data_name
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        save_path += "/" + title + ".png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plotKNN(data: np.ndarray, labels: np.ndarray, title: str, idx=None, dis_idx=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    label_set = set(labels.tolist())
    color = [
        "#900C3F",  # 紫红色
        "#556B2F",  # 暗绿色
        "#9370DB",  # 中紫色
        "#8B7355",  # 赭色
        "#A9A9A9",  # 暗灰色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
        "#A9A9A9",  # 暗灰色
        "#FFD700",  # 库金色
        '#606470', # 灰色
        '#6499E9',
        #'#db6400', # 淡蓝色
        '#F52F2F', #茄红色
        "#900C3F",  # 紫红色
        '#9DBDF5', # 橙色
        "#006400",  # 深绿色
        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
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
    fontSize = 20
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        colorNum = int(i % len(color))
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum], zorder=2)
    if idx is not None and dis_idx is not None:
        idxs = dis_idx[idx].reshape(-1, )
        i = 0
        while i < len(idxs) and dis_idx[idx, i] != -1:
            i += 1
        idxs = dis_idx[idx, :i].reshape(-1, )
        kk = len(idxs)
        for i in range(kk):
            plt.plot([data[idx, 0], data[idxs[i], 0]], [data[idx, 1], data[idxs[i], 1]],
                     color="#008080", zorder=1, linewidth=3)
        # for i in range(kk):
        #     plt.scatter(data[idxs[i], 0], data[idxs[i], 1], fontSize, color[12], lineform[0], zorder=2)
    if idx is not None:
        plt.scatter(data[idx, 0], data[idx, 1], fontSize + 10, color[13], lineform[0], zorder=2)
    # ax.set_ylim((-0.05, 0.85))
    # ax.set_xlim((-0.05, 1.05))
    ax.set_xticks([])
    ax.set_yticks([])
    # save_path = '../../data/aggSample/' + title + '.png'
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
