#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: Predict
@time: 2019/9/8 下午8:45
'''
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from classifier_dataset import classifier_25, SaveFile, LoadFile
from  re_sub import acc_regression

if __name__ == '__main__':
    path = ''
    path_cl = ''
    path_re = ''
    model_cl = tf.keras.models.load_model(filepath=path_cl)
    model_re = tf.keras.models.load_model(filepath=path_re)
    space_list = classifier_25(n=26)
    dataset = LoadFile(p=path)
    rng = np.random.RandomState(0)
    rng.shuffle(dataset)
    test_data = dataset[:6000, :] #根据导入数据标签改
    result_cl = model_cl.predict(x=[test_data[:, :4], test_data[:, 4:]], verbose=0)
    result_cl = np.argmax(a=result_cl, axis=1)
    result_inf = result_cl * 10 #将分类器执行后的结果对应到各个子分类空间中
    #regression
    result_re = model_re.predict(x=[test_data[:, :4], test_data[:, 4:]], verbose=0)
    re_guiyi = lambda x: (space_list[x][-1] - space_list[x][0]) * x + space_list[x][0]
    func = np.frompyfunc(re_guiyi, 1, 1)
    result_re = func(result_re)
    r_fin = result_cl + result_re
    #计算经过分类和回归器后的准确率
    T_list = [5, 3, 2, 1, 0.5]
    acc_list = (acc_regression(y_true=test_data[:, -1], y_pred=r_fin, Threshold=t) for t in T_list)
    print('整个回归模型的准确率与阈值选择对应关系为:')
    for t, acc in zip(T_list, acc_list):
        print('T:%s Acc:%s' % (t, acc))