import tensorflow as tf

class cnn_case(object):
    def __init__(self):
        return

    def test_cnn(self):
        '''
        滤波器 w*h*c*n
        输入节点 w*h*c
        输出节点 w*h*n
        :return:
        '''
        filter_weights = tf.get_variable(shape=[5,5,3,16], initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_biases = tf.get_variable(shape=[16], initializer=tf.constant_initializer(0.1))
        input_node = tf.get_variable(shape=[None, 5, 5, 3], initializer=tf.truncated_normal(stddev=0.1))
        conv = tf.nn.conv2d(input=input_node, filter=filter_weights, strides=[1,1,1,1], padding='SAME')
        bias = tf.nn.bias_add(conv, filter_biases)
        active_conv = tf.nn.relu(bias)

        pool = tf.nn.max_pool(value=active_conv, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')


        return
