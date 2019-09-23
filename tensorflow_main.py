from tensorflow_case.tensorflow import tensorflow_case
from tensorflow_case.tensor_neurons_case import bp_neural_networks
from tensorflow_bpnn.tbp_neural import tbp_neural
from mnist.load_mnist_data import mnist
from lenet_5.lenet_5 import le_net5
from optimizer.gradient_descent import gradient_descent
from rnn.lstm_imp import LSTM_IMPL
import argparse

def str2bool(v):
    if v == 'False':
        return False
    else:
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='lenet-5', type=str)
    parser.add_argument('--root_path', default='E:/data_warehouse/collision_warehouse/lstm', type=str)
    parser.add_argument('--file_key', default='label-lstm', type=str)
    parser.add_argument('--log_path', default='./lstm-log', type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--time_step', default=22, type=int)
    parser.add_argument('--out_num', default=1, type=int)
    parser.add_argument('--hide_num', default=(12,), type=int, nargs='+')
    parser.add_argument('--feature_col', default=('speed', 'acceleration', 'accelerationX', 'accelerationY', 'accelerationZ'), type=str, nargs='+')
    parser.add_argument('--label_col', default='flag', type=str)
    parser.add_argument('--time_step_column', default='flagtime', type=str)
    parser.add_argument('--train', default=True, type=str2bool)
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
        lstm = LSTM_IMPL(log_path=args.log_path,
                         file_key=args.file_key,
                         batch_size=args.batch_size,
                         time_step=args.time_step,
                         out_num=args.out_num,
                         hide_num=args.hide_num,
                         feature_col=args.feature_col,
                         label_col=args.label_col,
                         time_step_column=args.time_step_column)
        lstm.excute(args.root_path, args.train)

