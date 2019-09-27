import tensorflow as tf
import tensorflow.contrib.rnn as trnn
import numpy as np
import logging

class LSTM(object):
    def __init__(self, input_num, time_step, out_num, cell_unit, batch_size):
        '''
        :param input_num: 输入节点必须是特征个数
        :param time_step: 时间序列个数
        :param out_num: 输出层个数
        :param cell_unit: 隐藏层个数 list[10, 10, 10] 每个元素代表每个cell 中 激活函数输出维度
        :param batch_size:
        '''
        if not isinstance(cell_unit, tuple):
            raise RuntimeError('invalid cell_unit: the tyoe must be list ')
        self.cell_unit = cell_unit
        self.__input_num = input_num
        self.__out_num = out_num
        self.__time_step = time_step
        self.__batch_size = batch_size
        self.__logits, self.__y, self.__x = self._build_network()
        return

    def _init_weight(self, scope, input_num, out_num):
        with tf.variable_scope(scope):
            weight = tf.get_variable(name=scope+'wight', shape=[input_num, out_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable(name=scope+'bias', shape=[out_num], initializer=tf.constant_initializer(0))
        return weight, bias

    def _build_network(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.__time_step, self.__input_num])
        y = tf.placeholder(dtype=tf.float32, shape=[None, self.__out_num])

        input_weight = self._init_weight('input', self.__input_num, self.cell_unit[0])
        input_rnn = tf.reshape(x, [-1, self.__input_num])
        input_rnn = tf.matmul(input_rnn, input_weight[0]) + input_weight[1]

        input_rnn = tf.reshape(input_rnn, [-1, self.__time_step, self.cell_unit[0]])
        cell = None
        if len(self.cell_unit) > 1:
            cells = [tf.nn.rnn_cell.LSTMCell(unit) for unit in self.cell_unit]
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        else:
            cell = trnn.LSTMCell(self.cell_unit[0], state_is_tuple=True)
        init_state = cell.zero_state(self.__batch_size, tf.float32)
        rnn_output, state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, time_major=False)

        out_input = self._select_out(state, rnn_output, len(self.cell_unit)>1, select_state=True)

        output_weight = self._init_weight('output', self.cell_unit[-1], self.__out_num)
        logits = tf.matmul(out_input, output_weight[0]) + output_weight[1]
        return logits, y, x

    def _select_out(self, state, output, multi=False, select_state=False):
        '''
        :param state:  是lstm 最后一个cell的输出状态 因为lstm的输出有两个[ct,ht] 所以 state = [2, batch_size, cell_out_size]
        :param output:  把 它 转换成 [batch, out_num]*step 选用最后一个 rnn_output = tf.unstack(tf.transpose(output, [1,0,2]))
        '''
        out_input = None
        if select_state:
            if multi:
                out_input = state[len(self.cell_unit)-1][1]
            else:
                out_input = state[1]
        else:
            time_output = tf.transpose(output, [1, 0, 2])
            out_input = tf.reduce_mean(time_output, 0)

        return out_input

    def train(self, next_batch_data, optimize):
        '''
        x (batch_size, time_step, n_input)
        :return:
        '''
        optimize(next_batch_data, x=self.__x, logits=self.__logits, y=self.__y)
        return

    def predict(self, next_test_data, predict_mode):
        return predict_mode(next_test_data, x=self.__x, logits=self.__logits, y=self.__y)


