import tensorflow as tf
from functools import reduce
import numpy as np

class le_net5(object):
    def __init__(self,
                 input_shape,
                 full_shape,
                 filter_list,
                 filter_pool,
                 regularization_rate=0.01,
                 batch=100,
                 loses_name='loses'):
        '''
        :param input_shape: 第一个卷积层输入节点 [w,h,channel] 长,宽,通道
        :param full_shape: [n1,n2...] 全连接层每层的神经元个数
        :param filter_list: [[w, h, channel, depth, strides],]
                            depth 表示有多少组滤波器,
                            channel表示每组有多少个滤波器和输入得通道对应, 这个通道数和上一次计算结果有关
                            strides 步长
                            len(filter_list) 代表卷积层的个数
        :param filter_pool:[[w,h,strides],] 池化层的滤波器 len(filter_pool)同filter_list
        :param learn_rate: 全连接的学习率
        :param decay_rate:全连接的学习衰减率
        :param regularization_rate:全连接的正则化
        :param batch: 批处理大小
        :param moving_decay: 滑动平均衰减率
        :param loses_name:
        '''
        if not isinstance(full_shape, list):
            raise RuntimeError('filter type is not valid, need list type'.format(filter_list))
        if not isinstance(filter_list, list):
           raise RuntimeError('filter type is not valid, need list type'.format(filter_list))
        if not isinstance(filter_pool, list):
            raise RuntimeError('pool type is not valid, need list type'.format(filter_pool))
        if len(filter_list) != len(filter_pool):
            raise RuntimeError('pool and filter is not eq')

        self.__input_shape = input_shape
        self.__full_shape = full_shape
        self.__filter_list = filter_list
        self.__filter_pool = filter_pool
        self.__regularization_rate = regularization_rate
        self.__batch = batch
        self.__loses_name = loses_name
        return

    def __create_conv_layer(self, inputd, filter_shape, strides, scope):
        with tf.variable_scope(scope):
            filter_weights = tf.get_variable(name='weight', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            filter_biases = tf.get_variable(name='bias', shape=[filter_shape[-1]], initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(input=inputd, filter=filter_weights, strides=[1,strides,strides,1], padding='SAME')
            biase = tf.nn.bias_add(conv, filter_biases)
        return tf.nn.relu(biase)

    def __create_pool_layer(self, inputd, ksize, strides, scope):
        with tf.variable_scope(scope):
            pool = tf.nn.max_pool(value=inputd, ksize=[1, ksize[0], ksize[1], 1], strides=[1, strides, strides, 1], padding='SAME')
        return pool

    def __create_full_connect_layer(self, inputd, wshape, bshape, scope, regularizer=None, dropout=True, active=True):
        with tf.variable_scope(scope):
            weights = tf.get_variable(name='weight', shape=wshape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biase', shape=[bshape], initializer=tf.constant_initializer(0.1))
            if regularizer is not None:
                tf.add_to_collection(self.__loses_name, regularizer(weights))
            z = tf.matmul(inputd, weights) + biases
            a = z
            if active:
                a = tf.nn.relu(z)
                if dropout:
                    a = tf.nn.dropout(a, 0.5)
        return a

    def create_cnncreate_cnn(self):
        input_shape = self.__input_shape  #[batch, w, h, c]
        input_shape.insert(0, self.__batch)
        x = tf.placeholder(tf.float32, shape=input_shape, name='x-put')

        #卷积层
        conv = x
        for index in range(len(self.__filter_list)):
            filter_layer = self.__filter_list[index]
            pool_layer = self.__filter_pool[index]
            conv = self.__create_conv_layer(conv, filter_layer[0:-1], filter_layer[-1], 'conv'+str(index))
            conv = self.__create_pool_layer(conv, pool_layer[0:-1], pool_layer[-1], 'pool'+str(index))

        #全连接
        conv_shape = conv.get_shape().as_list() #[batch, w, h, c]
        full_input_node = reduce(lambda x, y: x*y, conv_shape[1:])
        full_input = tf.reshape(conv, shape=[conv_shape[0], full_input_node])
        full_node = self.__full_shape
        full_node.insert(0, full_input_node)
        regularizer = tf.contrib.layers.l2_regularizer(self.__regularization_rate)
        x_full = full_input
        for index in range(len(full_node)-2):
            x_full = self.__create_full_connect_layer(x_full,
                                                      [full_node[index], full_node[index+1]],
                                                      full_node[index+1], 'full'+str(index),
                                                      regularizer)
        logits = self.__create_full_connect_layer(x_full,
                                                  [full_node[-2], full_node[-1]],
                                                  full_node[-1], 'full-out',
                                                  regularizer, active=False)

        y = tf.placeholder(tf.float32, shape=[None, self.__full_shape[-1]], name='y-put')

        return x, logits, y
