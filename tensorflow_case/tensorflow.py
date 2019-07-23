import tensorflow as tf

class tensorflow_case(object):
    def __init__(self):
        return

    def test_graph(self):
        config = tf.ConfigProto(allow_soft_placement=True)

        a = tf.constant([1.2, 2.3], name='a')
        b = tf.constant([2.3, 3.0], name='b')
        result = a + b
        sess = tf.Session(config=config)
        with sess.as_default():
            print(result.eval())
        print(tf.Session().run(result))
        print(result)
        print(tf.get_default_graph)

        g1 = tf.Graph()
        with g1.as_default():
            tf.get_variable('v', shape=[1], initializer=tf.zeros_initializer())

        g2 = tf.Graph()
        with g2.as_default():
            tf.get_variable('v', shape=[1], initializer=tf.ones_initializer())

        with tf.Session(graph=g1) as sess:
            tf.initialize_all_variables().run()
            with tf.variable_scope('', reuse=True):
                print(sess.run(tf.get_variable('v')))

        with tf.Session(graph=g2) as sess:
            tf.initialize_all_variables().run()
            with tf.variable_scope('', reuse=True):
                print(sess.run(tf.get_variable('v')))

        return
