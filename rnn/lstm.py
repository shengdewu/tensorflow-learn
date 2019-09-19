import tensorflow as tf
import tensorflow.contrib.rnn as trnn
from log.log_configure import log_configure
import logging
from optimizer.gradient_descent import gradient_descent

class LSTM(object):
    def __init__(self, input_num, time_step, out_num, hide_num):
        '''
        :param input_num: 输入节点必须是特征个数
        :param time_step: 时间序列个数
        :param out_num: 输出层个数
        :param hide_num: 隐藏层个数 list[10, 23, 10] 每个元素代表没层rnn细胞个数
        '''
        if not isinstance(hide_num, list):
            raise RuntimeError('invalid hide_num: the tyoe must be list ')
        self.__hide_num = hide_num
        self.__input_num = input_num
        self.__out_num = out_num
        self.__time_step = time_step
        log_configure.init_log(log_name='lstm', log_path='./')
        return

    def _init_weight(self, scope, input_num, out_num):
        with tf.variable_scope(scope):
            weight = tf.get_variable(name=scope+'wight', shape=[input_num, out_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable(name=scope+'bias', shape=[out_num], initializer=tf.constant_initializer(0))
        return weight, bias

    def _build_network(self, x):
        input_weight = self._init_weight('input', self.__input_num, self.__hide_num[0])
        input_rnn = tf.reshape(x, [-1, self.__input_num])
        input_rnn = tf.matmul(input_rnn, input_weight[0]) + input_weight[1]
        input_rnn = tf.reshape(input_rnn, [-1, self.__time_step, self.__hide_num[0]])
        state = None
        for cell_num in self.__hide_num:
            cell = trnn.LSTMCell(cell_num, state_is_tuple=True)
            input_rnn, state = trnn.rnn.dynamic_rnn(cell, input_rnn)

        out_input = tf.reshape(input_rnn, [-1, self.__hide_num[-1]])
        outout_weight = self._init_weight('output', self.__hide_num[-1], self.__out_num)
        y = tf.matmul(out_input, outout_weight[0]) + outout_weight[1]
        return y, state

    def train(self, next_batch_data):
        '''
        x (batch_size, time_step, n_input)
        :return:
        '''
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.__time_step, self.__input_num])
        y = tf.placeholder(dtype=tf.float32, shape=[None, self.__time_step, self.__out_num])
        y1, s = self._build_network(x)
        optimize = gradient_descent()
        optimize.generalization_optimize(next_batch_data, x=x, logits=y1, y=y)
        return

