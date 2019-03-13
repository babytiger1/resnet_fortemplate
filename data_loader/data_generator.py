import numpy as np
import os
import gzip
import pickle
from tensorflow.examples.tutorials.mnist import input_data

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.mnist = input_data.read_data_sets(
            '../MNIST_data',
            one_hot=True)  # MNIST数据集所在路径
        # load data here

        # def load_dataset():
        #     url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        #     filename = 'mnist.pkl.gz'
        #     if not os.path.exists(filename):
        #         print("Downloading MNIST dataset...")
        #         urlretrieve(url, filename)
        #     with gzip.open(filename, 'rb') as f:
        #         data = pickle.load(f)
        #     X_train, y_train = data[0]
        #     X_val, y_val = data[1]
        #     X_test, y_test = data[2]
        #     X_train = X_train.reshape((-1, 1, 28, 28))
        #     X_val = X_val.reshape((-1, 1, 28, 28))
        #     X_test = X_test.reshape((-1, 1, 28, 28))
        #     y_train = y_train.astype(np.uint8)
        #     y_val = y_val.astype(np.uint8)
        #     y_test = y_test.astype(np.uint8)


    def next_batch(self):

        self.batch = self.mnist.train.next_batch(self.config.batch_size)

        yield self.batch[0], self.batch[1]

    def eval_data(self):
        return self.mnist.test.images, self.mnist.test.labels
