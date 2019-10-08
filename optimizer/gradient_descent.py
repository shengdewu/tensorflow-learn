import tensorflow as tf
import numpy as np
import logging

class gradient_descent(object):
    def __init__(self,
                 learn_rate=0.0001,
                 decay_rate=0.99,
                 moving_decay=0.99,
                 regularize_rate=0.01,
                 max_iter_times=50000,
                 mode_path=None,
                 update_mode_freq=100,
                 batch_size=100):
        '''
        :param learn_rate: 全连接的学习率
        :param decay_rate:全连接的学习衰减率
        :param moving_decay: 滑动平均衰减率
        :param regularize_rate:正则化率
        :param max_iter_times: 迭代次数
        :param mode_path:保存训练模型
        :param update_mode_freq: 模型保存频率
        :param batch_size: 批处理大小
        '''
        self.__learn_rate = learn_rate
        self.__decay_rate = decay_rate
        self.__moving_decay = moving_decay
        self.__regularize_rate = regularize_rate
        self.max_iter_times = max_iter_times
        self.__mode_path = mode_path
        self.update_mode_freq = update_mode_freq
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
            for i in range(self.max_iter_times):
                xs, ys = mnist.train.next_batch(input_shape[0])
                reshape_xs = np.reshape(xs, input_shape)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y: ys})
                if i % 1000 == 0:
                    print('after %d training step(s), loss on train batch is %g' % (step, loss_value))
        return

    def generalization_optimize(self, batch_data, x, logits, y, batch_size):
        global_step = tf.Variable(0, trainable=False)
        variable_averges = tf.train.ExponentialMovingAverage(self.__moving_decay, global_step)
        variable_averges_op = variable_averges.apply(tf.trainable_variables())

        regularizer = tf.contrib.layers.l2_regularizer(self.__regularize_rate)
        regularize_loss = [regularizer(v) for v in tf.trainable_variables() if len(v.shape) > 1]

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(regularize_loss)

        learn_rate = tf.train.exponential_decay(self.__learn_rate,
                                                global_step,
                                                self.__batch_size,
                                                self.__decay_rate, staircase=False)
        train_step = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss, global_step)

        with tf.control_dependencies([train_step, variable_averges_op]):
            train_op = tf.no_op(name='train')

        accuracy = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))[1]

        init_var = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        #持久化
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_var)
            max_accuracy = 0
            for i in range(self.max_iter_times):
                xs, ys = batch_data(self.__batch_size)
                if xs is None:
                    logging.warning('{} step(s) the data has been obtained!'.format(i))
                    break
                _, pred, accuracy_value, step = sess.run([train_op, logits, accuracy, global_step], feed_dict={x: xs, y: ys, batch_size:self.__batch_size})

                if i % self.update_mode_freq == 0:
                    if self.__mode_path is not None:
                        if max_accuracy < accuracy_value:
                            max_accuracy = accuracy_value
                            saver.save(sess, save_path=self.__mode_path)
                    #print('predict {}\n label {}'.format(np.argmax(pred, axis=1), np.argmax(ys, axis=1)))
                    print('after %d training step(s), accuracy on train batch is %g' % (step, accuracy_value))
            # test_xs, test_ys = batch_data(batch_size=None, train=False)
            # self._eval(x, test_xs, y, test_ys, accuracy, logits, sess, batch_size)
        return

    def _eval(self, x, xs, y, ys, accuracy, logits, sess, batch_size):
        acc, pred = sess.run([accuracy, logits], feed_dict={x: xs, y: ys, batch_size:xs.shape[0]})
        pred = np.argmax(pred, axis=1)
        label = np.argmax(ys, axis=1)

        # tp = 0
        # fn = 0
        # fp = 0
        # for p, l in zip(pred, label):
        #     if p == l and p == 1:
        #         tp += 1
        #     if p == 1 and l == 0:
        #         fp += 1
        #     if p == 0 and l == 1:
        #         fn += 1
        #
        # precision = np.inf
        # if tp + fp != 0:
        #     precision = tp / (tp + fp)
        # recall = np.inf
        # if tp + fn != 0:
        #     recall = tp / (tp + fn)
        #
        # print('optimize eval, test accuracy is {}, [label:{}-pred:{}],precision={}, recall={}'.format(acc,label, pred, precision, recall))
        return pred, label

    def generalization_predict(self, next_data, x, logits, y, batch_size):
        predict_result = None
        accuracy = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(logits, 1))[1]

        self.__batch_size = 1
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('load model from {}'.format(self.__mode_path))
            saver.restore(sess, self.__mode_path)
            sess.run(tf.local_variables_initializer())  #tf.global_variables_initializer() 不用初始化，模型已经初始化了，否则会造成严重后果

            xs, ys = next_data(batch_size=self.__batch_size, train=False)
            while xs is not None:
                pred, label = self._eval(x, xs, y, ys, accuracy, logits, sess, batch_size)

                pc = np.reshape(pred, (pred.shape[0], 1))
                label = np.reshape(label, (label.shape[0], 1))
                label = np.concatenate((label, pc), axis=1)

                if predict_result is None:
                    predict_result = label
                else:
                    predict_result = np.concatenate((predict_result, label), axis=0)

                xs, ys = next_data(self.__batch_size, train=False)
        return predict_result

