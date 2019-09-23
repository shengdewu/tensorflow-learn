import tensorflow as tf
import numpy as np
import logging

class gradient_descent(object):
    def __init__(self,
                 learn_rate=0.001,
                 decay_rate=0.99,
                 moving_decay=0.99,
                 train_step=50000,
                 mode_path=None):
        '''
        :param learn_rate: 全连接的学习率
        :param decay_rate:全连接的学习衰减率
        :param moving_decay: 滑动平均衰减率
        :param train_step: 迭代次数
        :param mode_path:保存训练模型
        '''
        self.__learn_rate = learn_rate
        self.__decay_rate = decay_rate
        self.__moving_decay = moving_decay
        self.__train_step = train_step
        self.__mode_path = mode_path
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
        cross_entropy = tf.square(tf.reshape(logits, [-1]) - tf.reshape(y,[-1]))
        loss = tf.reduce_mean(cross_entropy)
        learn_rate = tf.train.exponential_decay(self.__learn_rate,
                                                0,
                                                logits.shape[0],
                                                self.__decay_rate, staircase=False)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss, global_step)

        with tf.control_dependencies([train_step, variable_averges_op]):
            train_op = tf.no_op(name='train')

        #持久化
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print_log_num = self.__train_step // 100
            tf.global_variables_initializer().run()
            for i in range(self.__train_step):
                xs, ys = batch_data()
                logging.debug('xs.shape {}, x.shape {}, ys.shape {}, y.shape{}\n'.format(xs.shape, x.shape, ys.shape, y.shape))
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y: ys})
                if i % print_log_num == 0:
                    if self.__mode_path is not None:
                        saver.save(sess, save_path=self.__mode_path)
                    print('after %d training step(s), loss on train batch is %g' % (step, loss_value))
        return

    def generalization_predict(self, next_data, x, logits, y):
        predict_result = None
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, self.__mode_path)

            test_data = next_data()
            while test_data is not None:
                xs = test_data[0]
                ys = test_data[1]
                p = sess.run(logits, feed_dict={x:xs, y:ys})
                x1 = np.reshape(xs, [-1])
                y1 = np.reshape(ys, [-1])
                p1 = np.reshape(p, [-1])

                loss = np.sum(y1 == p1)
                loss = loss / y1.shape[0]
                print('test {}'.format(loss))
                #
                # td = np.concatenate((x1, y1), axis=1)
                # if predict_result is None:
                #     predict_result = np.concatenate((td, p1), axis=1)
                # else:
                #     t = np.concatenate((td, p1), axis=1)
                #     predict_result = np.concatenate((predict_result, t), axis=0)

                test_data = next_data()
        return predict_result

