#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: classifier_dataset
@time: 2019/9/8 下午8:45
'''
import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
##################最优半径区间划分为25个子区间###################

def LoadFile(p):
    '''
    读取文件
    :param p: 数据集绝对路径
    :return: 数据集
    '''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def SaveFile(data, savepickle_p):
        '''
        存储整理好的数据
        :param data: 待存储数据
        :param savepickle_p: pickle后缀文件存储绝对路径
        :return: None
        '''
        if not os.path.exists(savepickle_p):
            with open(savepickle_p, 'wb') as file:
                pickle.dump(data, file)

def classifier_25(n):
    '''
    细化最优半径区间
    :param n: 区间个数
    :return: 区间划分列表
    '''
    space = np.linspace(0, 50, n)
    space_sub1 = space[0:-1]
    space_sub2 = space[1:]
    space_sub2[-1] = 52
    space_sub1 = list(space_sub1)
    space_sub2 = list(space_sub2)
    space = zip(space_sub1, space_sub2)
    return list(space)

def dataset_cl(dataset, space):
    '''
    将数据集标签按照最优半径所属于的范围进行划分
    :param dataset: 数据集
    :param space: 半径划分区间列表
    :return: 处理后的数据
    '''
    dataset_return = np.array([0])
    dataset_pd = pd.DataFrame(data=dataset, columns=['F' + str(i) for i in range(dataset.shape[-1]-1)] + ['r'])
    for inf, sup in space:
        dataset_sub_pd = dataset_pd.loc[dataset_pd['r'] >= inf]
        dataset_sub_pd = dataset_sub_pd.loc[dataset_sub_pd['r'] < sup]
        if dataset_sub_pd.values.shape[0] == 0: continue
        dataset_sub_pd['r'] = space.index((inf, sup))
        dataset_return = dataset_sub_pd.values if dataset_return.any() == 0 else \
            np.vstack((dataset_return, dataset_sub_pd.values))
    return dataset_return

def checkclassifier(vector):
    '''
    对输入数据向量进行各个数量类别统计
    :param vector: 待统计数据向量
    :return: None
    '''
    statistic = Counter(vector)
    statistic = sorted(statistic.items(), key=lambda x: x[0])
    for key, value in statistic:
        print('%s: %s' % (key, value))

def dataset_junheng(dataset, number):
    '''
    对所有半径类别数据进行数据均衡化
    :param dataset: 数据集
    :param number: 各类半径数量
    :return: 处理后数据
    '''
    dataset_pd = pd.DataFrame(data=dataset, columns=[str(i) for i in range(25)])
    dataset_return = np.array([0])
    for i in range(20):
        sub_dataset = dataset_pd.loc[dataset_pd['24'] == i]
        if sub_dataset.values.shape[0] > number:
            dataset_return = sub_dataset.values[:number, :] if dataset_return.any() == 0 else \
                np.vstack((dataset_return, sub_dataset.values[:number, :]))
        elif sub_dataset.values.shape[0] and sub_dataset.values.shape[0] < number:
            judge = number % sub_dataset.values.shape[0]
            num = number // sub_dataset.values.shape[0]
            if judge != 0:
                num += 1
            dataset_sub2000 = sub_dataset.values
            for i in range(num-1):
                dataset_sub2000 = np.vstack((dataset_sub2000, sub_dataset.values))
            dataset_sub2000 = dataset_sub2000[:number, :]
            dataset_return = dataset_sub2000 if dataset_return.any() == 0 else \
                np.vstack((dataset_return, dataset_sub2000))
    return dataset_return

def guiyi(dataset):
    '''
    对带标签的数据集进行特征归一化
    :param dataset: 带标签的数据集
    :return: 归一化后的特征/标签矩阵
    '''
    feature_min = np.min(dataset[:, :-1], axis=0)
    feature_max = np.max(dataset[:, :-1], axis=0)
    feature_guiyi = (dataset[:, :-1] - feature_min) / (feature_max - feature_min)
    dataset_guiyi = np.hstack((feature_guiyi, dataset[:, -1][:, np.newaxis]))
    return dataset_guiyi

def fft_transformer(dataset, N):
    '''
    对矩阵中各行按照指定点数做FFT变换
    :param dataset: 待处理矩阵
    :param N: 变换后点数
    :return: 处理后矩阵
    '''
    fft_abs = np.abs(np.fft.fft(a=dataset, n=N, axis=1))
    return fft_abs


if __name__ == '__main__':
    space = classifier_25(21)
    # print(space)
    p = '/home/xiaosong/oldenburg相关数据/data_oldenburg/OLDENBURG_all.pickle'
    dataset = LoadFile(p)
    # print(dataset.shape)
    print(Counter(dataset[:, -1]))
    dataset_cl25 = dataset_cl(dataset=dataset, space=space)
    print(dataset_cl25.shape)
    checkclassifier(dataset_cl25[:, -1])
    dataset_cl25_2000 = dataset_junheng(dataset=dataset_cl25, number=2000)
    checkclassifier(dataset_cl25_2000[:, -1])
    print(dataset_cl25_2000.shape)
    dataset_4feature, dataset_dense, label = dataset_cl25_2000[:, :4], dataset_cl25_2000[:, 4:-1], \
                                             dataset_cl25_2000[:, -1][:, np.newaxis]
    dataset_fft = fft_transformer(dataset_dense, 100)
    dataset = np.hstack((dataset_4feature, dataset_fft, label))
    dataset_guiyi = guiyi(dataset)
    print(dataset_guiyi.shape)
    SaveFile(data=dataset_guiyi, savepickle_p='/home/xiaosong/桌面/oldenburg_cl20.pickle')
    print(np.max(dataset_guiyi, axis=0))