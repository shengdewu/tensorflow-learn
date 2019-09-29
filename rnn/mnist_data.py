from mnist.load_mnist_data import mnist
import pandas as pd
import numpy as np
import logging
import os
import re

class mnist_frame(object):
    def __init__(self, mnist_path='mnist/data/'):
        self.__data = mnist.load_mnist_data(mnist_path)
        return

    def next_batch(self, batch_size=None, train=True):
        xs = None
        ys = None
        if train:
            xs, ys = self.__data.train.next_batch(batch_size)
            xs = np.reshape(xs, (batch_size, 28, 28))
        else:
            xs = self.__data.test.images
            ys = self.__data.test.labels
            xs = np.reshape(xs, (xs.shape[0], 28, 28))
        return xs, ys



