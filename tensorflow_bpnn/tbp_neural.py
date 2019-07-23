import tensorflow as tf

class tbp_neural(object):
    def __init__(self,
                 input_node, #输入节点个数
                 output_node, #输出节点个数
                 hide_node, #隐藏层个数 必须是list
                 learn_rate=0.001, #学习率
                 decay_rate=0.99, #学习率衰减率
                 batch_size=100,#批量数据个数 越大越接近梯度下降，越小越接近随机梯度下降
                 regularization_rate=0.0001,#正则化系数
                 train_step=30000,#迭代次数
                 moving_avgerage_decay=0.99, #滑动平均衰减率
                 num_example = None  #样本总数
                ):
        self.__input_node = input_node
        self.__output_node = output_node
        if not isinstance(hide_node, list):
            raise RuntimeError('hide layer param is not valid, need list type')
        self.__hide_node = hide_node
        self.__learn_rate = learn_rate
        self.__decay_rate = decay_rate
        self.__batch_size = batch_size
        self.__regularization_rate = regularization_rate
        self.__train_step = train_step
        self.__moving_avgerage_decay = moving_avgerage_decay
        self.__num_example = num_example
        return

    def __caculate_output(self, input, weight, baise, sigmod=True, moving_avg=None):
        a = None
        if sigmod:
            if moving_avg is None:
                a = tf.nn.relu(tf.matmul(input, weight)+baise)
            else:
                a = tf.nn.relu(tf.matmul(input, moving_avg.average(weight))+moving_avg.average(baise))
        else:
            if moving_avg is None:
                a = tf.matmul(input, weight)+baise
            else:
                a = tf.matmul(input, moving_avg.average(weight))+moving_avg.average(baise)
        return a

    def __init_param(self, regularizer=None):
        weights = []
        biases = []
        layers = [self.__input_node]
        layers.extend(self.__hide_node)
        layers.append(self.__output_node)
        for l1, l2 in zip(layers[:-1], layers[1:]):
            w = tf.Variable(initial_value=tf.truncated_normal(shape=[l1, l2], stddev=0.1))
            b = tf.Variable(initial_value=tf.constant(0.1, shape=[l2]))
            weights.append(w)
            biases.append(b)
            if regularizer:
                tf.add_to_collection('losses', regularizer(w))
        return weights, biases

    def train(self, mnist, mode_path=None):
        x = tf.placeholder(tf.float32, shape=(None, self.__input_node), name='x-input')
        y = tf.placeholder(tf.float32, shape=(None, self.__output_node), name='y-input')

        regularizer = tf.contrib.layers.l2_regularizer(self.__regularization_rate)
        weights, biases = self.__init_param(regularizer)
        a = x
        for w, b in zip(weights[0:-1], biases[0:-1]):
            a = self.__caculate_output(a, w, b)
        y_ = self.__caculate_output(a, weights[-1], biases[-1], False)

        #所有训练参数上使用滑动平均参数，抑制突变
        num_updates = tf.Variable(initial_value=0, trainable=False)
        moving_avg = tf.train.ExponentialMovingAverage(self.__decay_rate, num_updates)
        variables_averages_op = moving_avg.apply(tf.trainable_variables())

        a_avg = x
        for w, b in zip(weights[0:-1], biases[0:-1]):
            a_avg = self.__caculate_output(a_avg, w, b, True, moving_avg)
        y_avg_ = self.__caculate_output(a_avg, weights[-1], biases[-1], False, moving_avg)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.argmax(y, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # regularizer = tf.contrib.layers.l2_regularizer(self.__regularization_rate)
        # regularization = regularizer(weights[0])
        # for i in range(1, len(weights), 1):
        #     regularization += regularizer(weights[i])

        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

        #loss = cross_entropy_mean
        learning_rate = tf.train.exponential_decay(self.__learn_rate,
                                                   num_updates,
                                                   mnist.train.num_examples /self.__batch_size,
                                                   self.__decay_rate)

        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, num_updates)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # with tf.control_dependencies([train_step]):
        #     train_op = tf.no_op(name='train')

        correct_prediction = tf.equal(tf.argmax(y_avg_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #持久化
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            validate_feed = {x:mnist.validation.images, y:mnist.validation.labels}
            test_feed = {x:mnist.test.images, y:mnist.test.labels}
            for i in range(self.__train_step):
                if i % 1000 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print('after %d training step(s), validation accuracy using average model is %g'%(i, validate_acc))
                    saver.save(sess, save_path=mode_path+'bp')

                xs, ys = mnist.train.next_batch(self.__batch_size)
                sess.run(train_op, feed_dict={x:xs, y:ys})

            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print('after %d training step(s), test accuracy using average model is %g' % (self.__train_step, test_acc))
        return

