#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: re_sub
@time: 2019/9/8 下午8:46
'''
import tensorflow as tf
import numpy as np
from cl25 import spliting, LoadFile, SaveFile, input
def acc_regression(Threshold, y_true, y_pred):
    '''
    回归精确度（预测值与实际值残差在阈值范围内的数量/样本总数）
    :param Threshold: 预测值与实际值之间的绝对值之差阈值
    :param y_true: 样本实际标签
    :param y_pred: 样本预测结果
    :return: 精确率，type: ndarray
    '''
    # 残差布尔向量
    is_true = np.abs(y_pred - y_true) <= Threshold
    is_true_cast = np.where(is_true, 1, 0)
    acc_rate_regression = np.sum(is_true_cast) / is_true_cast.shape[0]
    return acc_rate_regression

def R_regression():
    '''
    后期回归模型: cnn+lstm+dnn
    :return: 回归模型
    '''
    with tf.name_scope('input'):
        input1 = tf.keras.layers.Input(shape=(4, ), name='input1')
        input2 = tf.keras.layers.Input(shape=(100,), name='input2')
    with tf.name_scope('cnn'):
        layer = tf.keras.layers.Reshape(target_shape=[10, 10, 1], name='reshape')(input2)
        layer = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(), name='conv1')(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(layer)
        layer = tf.keras.layers.BatchNormalization(name='bn1')(layer)
        layer = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(), name='conv2')(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool2')(layer)
        layer = tf.keras.layers.Flatten(name='flat1')(layer)
    with tf.name_scope('rnn'):
        # rnn层
        layer = tf.keras.layers.Reshape(target_shape=[24, 24], name='x_lstm')(layer)
        layer = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='lstm1')(layer)
        layer = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='lstm2')(layer)
        layer = tf.keras.layers.Flatten(name='flat2')(layer)
    with tf.name_scope('dnn'):
        # dnn层
        def concat(inputs):
            return tf.concat(values=inputs, axis=1)
        layer = tf.keras.layers.Lambda(concat, name='concat1')(inputs=[input1, layer])
        layer = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                      bias_initializer=tf.keras.initializers.TruncatedNormal(), name='x_fc1')(layer)
        layer = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')(layer)
        layer = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                      bias_initializer=tf.keras.initializers.TruncatedNormal(), name='x_fc2')(layer)
        layer = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')(layer)
        layer = tf.keras.layers.Dense(units=1, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                      bias_initializer=tf.keras.initializers.TruncatedNormal(), name='x_fc3')(layer)
    model = tf.keras.Model(inputs=[input1, input2], outputs=layer)
    return model

def graph_re(dataset, save_path):
    ''''''
    r_regression = R_regression()
    optimizer = tf.keras.optimizers.SGD(lr=1e-2)
    r_regression.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error)
    train_data, test_data = spliting(dataset, 6600)
    flag = 0
    for  epoch in range(20000):
        for train_data_batch in input(dataset=train_data, batch_size=500):
            loss_train = r_regression.train_on_batch(x=[train_data_batch[:, :4], train_data_batch[:, 4:-1]],
                                                        y=train_data_batch[:, -1])
            if epoch % 100 == 0 and flag == 0:
                print('第%s轮后训练集损失函数值为: %s' % (epoch, loss_train))
                flag = 1
        if epoch % 100 == 0:
            r_predict = r_regression.predict(x=[test_data[:, :4], test_data[:, 4:-1]], verbose=0)
            acc1 = acc_regression(Threshold=0.5, y_true=test_data[:, -1][:, np.newaxis], y_pred=r_predict)
            # acc2 = acc_regression(Threshold=0.2, y_true=test_data[:, -1][:, np.newaxis], y_pred=r_predict)
            # acc3 = acc_regression(Threshold=0.1, y_true=test_data[:, -1][:, np.newaxis], y_pred=r_predict)
            # print('T1=%s, acc1=%s  T2=%s, acc2=%s, T3=%s, acc3=%s' % (0.3, acc1, 0.2, acc2, 0.1, acc3))
            print('T1=%s, acc1=%s' % (0.5, acc1))
            if acc1 > 0.99: break
        flag = 0
    r_regression.save(save_path)

if __name__ == '__main__':
    path = '/home/xiaosong/桌面/oldenburg_regression_sub.pickle'
    save_path = '/home/xiaosong/桌面/graph_cl_re_old/graph_re.h5'
    dataset = LoadFile(p=path)
    graph_re(dataset=dataset, save_path=save_path)
    #测试模型
    model = tf.keras.models.load_model(save_path)
    train_data, test_data = spliting(dataset, 6600)
    r_predict = model.predict(x=[test_data[:, :4], test_data[:, 4:-1]], verbose=0)
    acc = acc_regression(Threshold=0.5, y_true=test_data[:, -1][:, np.newaxis], y_pred=r_predict)
    print('测试集准确率为: %s' % acc)