import tensorflow as tf

class gradient_descent(object):
    def __init__(self):
        return

    def optimize(self, mnist, logits, y, x):
        cros_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.arg_max(y, 1))

        return
