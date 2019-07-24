import tensorflow as tf

class le_net5(object):
    def __init__(self):
        return

    def __create_conv_layer(self, inputd, filter_shape, strides):
        filter_weights = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_biases = tf.get_variable(shape=[filter_shape[-1]], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input=inputd, filter=filter_weights, strides=[1,strides,strides,1], padding='VALID')
        biase = tf.nn.bias_add(conv, filter_biases)
        return tf.nn.relu(biase)

    def __create_pool_layer(self, inputd, ksize, strides):
        pool = tf.nn.max_pool(input=inputd, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='VALID')
        return pool

    def __create_full_connect_layer(self, inputd, wshape, bshape):
        weights = tf.get_variable(shape=wshape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(shape=[bshape], initializer=tf.constant_initializer(0.1))
        return tf.nn.relu(tf.matmul(inputd, weights) + biases)

    def train(self):

        return
