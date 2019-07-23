from tensorflow.examples.tutorials.mnist import input_data

class mnist(object):
    @staticmethod
    def load_mnist_data(path):
        mnist = input_data.read_data_sets(path, one_hot=True)
        print('Training data size: {0}'.format(mnist.train.num_examples))
        print('Validating data size: {0}'.format(mnist.validation.num_examples))
        print('Testing data size: {0}'.format(mnist.test.num_examples))
        print('Example training data: {0}'.format(mnist.train.images[0]))
        print('Example training label: {0}'.format(mnist.train.labels[0]))
        return mnist
