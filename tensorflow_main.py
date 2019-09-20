from tensorflow_case.tensorflow import tensorflow_case
from tensorflow_case.tensor_neurons_case import bp_neural_networks
from tensorflow_bpnn.tbp_neural import tbp_neural
from mnist.load_mnist_data import mnist
from lenet_5.lenet_5 import le_net5
from optimizer.gradient_descent import gradient_descent
from rnn.lstm_imp import LSTM_IMPL
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='lenet-5', type=str)
    args = parser.parse_args()
    opt = args.option

    if opt == 'case':
        #case
        tensorflow_test = tensorflow_case()
        tensorflow_test.test_graph()

        tensorflow_nn = bp_neural_networks('./')
        tensorflow_nn.bp_nn()

    if opt == 'bp':
        #bp
        mnist = mnist.load_mnist_data('mnist/data/')
        neural_network = tbp_neural(input_node=784,
                                    output_node=10,
                                    hide_node=[500],
                                    batch_size=100,
                                    learn_rate=0.8,
                                    decay_rate=0.99,
                                    regularization_rate=0.0001,
                                    train_step=30000,
                                    moving_avgerage_decay=0.99)
        neural_network.train(mnist, 'tensorflow_case/mode/')

    if opt == 'lenet-5':
        le_net_5_mode = le_net5(input_shape=[28, 28, 1],
                                full_shape=[512, 10],
                                filter_list=[[5,5,1,32,1],[5,5,32,64,1]], #注意通道的赋值
                                filter_pool=[[2,2,2],[2,2,2]],
                                batch=100,
                                regularization_rate=0.00001)
        mnist = mnist.load_mnist_data('mnist/data/')
        x, logits, y = le_net_5_mode.create_cnncreate_cnn()
        optimizer_mode = gradient_descent()
        optimizer_mode.optimize(mnist, x, logits, y)

    if opt == 'lstm':
        lstm = LSTM_IMPL()
        lstm.excute('E:/data_warehouse/collision_warehouse/lstm/lstm-label.csv')

