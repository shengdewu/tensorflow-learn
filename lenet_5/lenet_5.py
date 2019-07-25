import tensorflow as tf
from functools import reduce

class le_net5(object):
    def __init__(self, input_node,input_shape, full_shape, filter_list, filter_pool, learn_rate=0.001, decay_rate=0.99, regularization_rate=0.01, batch=2000):
        '''
        :param input_node: 输入节点
        :param input_shape: 第一个卷积层输入节点 [w,h,channel] 长,宽,通道
        :param full_shape: [n1,n2...] 全连接层每层的神经元个数
        :param filter_list: [[w, h, channel, depth, strides],]
                            depth 表示有多少组滤波器,channel表示每组有多少个滤波器和输入得通道对应,
                            strides 步长
                            len(filter_list) 代表卷积层的个数
        :param filter_pool:[[w,h,strides],] 池化层的滤波器 len(filter_pool)同filter_list
        :param learn_rate: 全连接的学习率
        :param decay_rate:全连接的学习衰减率
        :param regularization_rate:全连接的正则化
        :param batch: 批处理大小
        '''
        if not isinstance(full_shape, list):
            raise RuntimeError('filter type is not valid, need list type'.format(filter_list))
        if not isinstance(filter_list, list):
           raise RuntimeError('filter type is not valid, need list type'.format(filter_list))
        if not isinstance(filter_pool, list):
            raise RuntimeError('pool type is not valid, need list type'.format(filter_pool))
        if len(filter_list) != len(filter_pool):
            raise RuntimeError('pool and filter is not eq')

        self.__input_node = input_node
        self.__input_shape = input_shape
        self.__full_shape = full_shape
        self.__filter_list = filter_list
        self.__filter_pool = filter_pool
        self.__learn_rate = learn_rate
        self.__decay_rate = decay_rate
        self.__regularization_rate = regularization_rate
        self.__batch = batch
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

    def __create_full_connect_layer(self, inputd, wshape, bshape, regularizer=None, dropout=False, active=True):
        weights = tf.get_variable(shape=wshape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(shape=[bshape], initializer=tf.constant_initializer(0.1))
        if regularizer is not None:
            tf.add_to_collection('loses', regularizer(weights))
        z = tf.matmul(inputd, weights) + biases
        a = z
        if active:
            a = tf.nn.relu(z)
            if dropout:
                a = tf.nn.dropout(a, 0.5)
        return a

    def create_mode(self, mnist):
        input_shape = self.__input_shape #[batch, w, h, c]
        input_shape.insert(0, self.__batch)
        x = tf.placeholder(tf.float32, shape=[None, self.__input_node], name='x-put')

        #卷积层
        conv = x
        for index in range(len(self.__filter_list)):
            filter_layer = self.__filter_list[index]
            pool_layer = self.__filter_pool[index]
            conv = self.__create_conv_layer(conv, filter_layer[0:-1], filter_layer[-1], 'conv'+str(index))
            conv = self.__create_pool_layer(conv, pool_layer[0:-1], pool_layer[-1], 'pool'+str(index))

        #全连接
        conv_shape = tf.shape(conv) #[batch, w, h, c]
        full_input_node = reduce(lambda x, y: x*y, conv_shape[1:])
        full_input = tf.reshape(conv, shape=[conv_shape[0], full_input_node])
        full_node = self.__full_shape
        full_node.insert(0, full_input_node)
        regularizer = tf.contrib.layers.l2_regularizer(self.__regularization_rate)
        x = full_input
        for index in range(len(full_node)-2):
            x = self.__create_full_connect_layer(x,
                                                 [full_node[index], full_node[index+1]],
                                                 full_node[index+1], regularizer)
        y_ = self.__create_full_connect_layer(x,
                                              [full_node[-2], full_node[-1]],
                                              full_node[-1], regularizer, False)

        #滑动平均
        global_step = tf.Variable(0, trainable=False)
        variable_averges = tf.train.ExponentialMovingAverage(0.5, global_step)
        variable_averges_op = variable_averges.apply(tf.trainable_variables())

        y = tf.placeholder(tf.float32, shape=self.__full_shape[-1], name='y-put')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.arg_max(y, 1))
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('loses'))

        learn_rate = tf.train.exponential_decay(self.__learn_rate, 0, mnist.train.num_examples /self.__batch, self.__decay_rate)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss, global_step)

        with tf.control_dependencies([train_step, variable_averges_op]):
            train_op = tf.no_op(name='train')

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for i in range(5000):
                xs, ys = mnist.next_batch(self.__batch)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y:ys})
                if i % 1000 == 0:
                    print('after %d training step(s), loss on train batch is %g' % (step, loss_value))
        return y_
