import tensorflow as tf
import numpy as np

class gradient_descent(object):
    def __init__(self,
                 learn_rate=0.001,
                 decay_rate=0.99,
                 moving_decay=0.99,
                 train_step=50000):
        '''
        :param learn_rate: 全连接的学习率
        :param decay_rate:全连接的学习衰减率
        :param moving_decay: 滑动平均衰减率
        :param train_step: 迭代次数
        '''
        self.__learn_rate = learn_rate
        self.__decay_rate = decay_rate
        self.__moving_decay = moving_decay
        self.__train_step = train_step
        return

    def optimize(self, mnist, x, logits, y, loses_name='loses'):
        input_shape = x.get_shape().as_list()
        # 滑动平均
        global_step = tf.Variable(0, trainable=False)
        variable_averges = tf.train.ExponentialMovingAverage(self.__moving_decay, global_step)
        variable_averges_op = variable_averges.apply(tf.trainable_variables())

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.arg_max(y, 1))
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection(loses_name))

        learn_rate = tf.train.exponential_decay(self.__learn_rate,
                                                0,
                                                mnist.train.num_examples / input_shape[0],
                                                self.__decay_rate, staircase=False)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss, global_step)

        with tf.control_dependencies([train_step, variable_averges_op]):
            train_op = tf.no_op(name='train')

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for i in range(self.__train_step):
                xs, ys = mnist.train.next_batch(input_shape[0])
                reshape_xs = np.reshape(xs, input_shape)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y: ys})
                if i % 1000 == 0:
                    print('after %d training step(s), loss on train batch is %g' % (step, loss_value))
        return
