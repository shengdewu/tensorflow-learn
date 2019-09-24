import tensorflow as tf
import numpy as np
import logging

class gradient_descent(object):
    def __init__(self,
                 learn_rate=0.001,
                 decay_rate=0.99,
                 moving_decay=0.99,
                 train_step=50000,
                 mode_path=None,
                 save_freq=100,
                 batch_size=100):
        '''
        :param learn_rate: 全连接的学习率
        :param decay_rate:全连接的学习衰减率
        :param moving_decay: 滑动平均衰减率
        :param train_step: 迭代次数
        :param mode_path:保存训练模型
        :param save_freq: 模型保存频率
        :param batch_size: 批处理大小
        '''
        self.__learn_rate = learn_rate
        self.__decay_rate = decay_rate
        self.__moving_decay = moving_decay
        self.__train_step = train_step
        self.__mode_path = mode_path
        self.__save_freq = save_freq
        self.__batch_size = batch_size
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

    def generalization_optimize(self, batch_data, x, logits, y):
        global_step = tf.Variable(0, trainable=False)
        variable_averges = tf.train.ExponentialMovingAverage(self.__moving_decay, global_step)
        variable_averges_op = variable_averges.apply(tf.trainable_variables())

        logging.debug('logits.shape {}, y.shape {} ymax.shape {}'.format(logits.shape, y.shape, tf.arg_max(y, 1).shape))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.arg_max(y, 1))
        loss = tf.reduce_mean(cross_entropy)
        learn_rate = tf.train.exponential_decay(self.__learn_rate,
                                                global_step,
                                                logits.shape[0],
                                                self.__decay_rate, staircase=False)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss, global_step)

        with tf.control_dependencies([train_step, variable_averges_op]):
            train_op = tf.no_op(name='train')

        #持久化
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            min_err = 1e10
            min_step = 0
            for i in range(self.__train_step):
                xs, ys = batch_data(self.__batch_size)
                if xs is None:
                    logging.warning('{} step(s) the data has been obtained!'.format(i))
                    break
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y: ys})
                if i % self.__save_freq == 0:
                    if self.__mode_path is not None:
                        if min_err > loss_value:
                            min_err = loss_value
                            min_step = step
                            saver.save(sess, save_path=self.__mode_path)
                    print('after %d training step(s), loss on train batch is %g' % (step, loss_value))
            print('optimize %d training step(s), loss on train batch is %g' % (min_step, min_err))
        return

    def generalization_predict(self, next_data, x, logits):
        predict_result = None
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, self.__mode_path)

            xs, ys = next_data(self.__batch_size)
            while xs is not None:
                p = sess.run(logits, feed_dict={x:xs})
                pc = np.argmax(p, axis=1)
                pc = np.reshape(pc, (pc.shape[0], 1))
                y = np.argmax(ys, axis=1)
                y = np.reshape(y, (y.shape[0], 1))
                y1 = np.concatenate((y, pc), axis=1)

                if predict_result is None:
                    predict_result = y1
                else:
                    predict_result = np.concatenate((predict_result, y1), axis=0)

                xs, ys = next_data(self.__batch_size)
        return predict_result

