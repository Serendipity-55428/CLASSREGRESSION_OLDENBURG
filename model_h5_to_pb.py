#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: model_h5_to_pb
@time: 2019/9/8 下午8:46
'''
from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util,graph_io
from tensorflow.python.tools import import_pb_to_tensorboard

def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    '''
    将h5格式的计算图转化为pb格式的计算图
    :param h5_model: h5格式的计算图
    :param output_dir: pb格式保存路径
    :param model_name: 生成的pb文件模型路径
    :param out_prefix: 生成的pb文件中输出节点前缀
    :param log_tensorboard: 是否对计算图进行可视化的标记
    :return:
    '''
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

if  __name__ == '__main__':
    # pb格式模型保存路径
    output_dir = ''
    # h5模型路径
    path = ''
    # 生成的pb文件模型路径
    path_pb = ''
    h5_model = load_model(filepath=path)
    h5_to_pb(h5_model, output_dir=output_dir, model_name=path_pb)