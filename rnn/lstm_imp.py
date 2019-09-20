from rnn.data_tool import data_frame
from rnn.lstm import LSTM
from optimizer.gradient_descent import gradient_descent
from log.log_configure import log_configure

class LSTM_IMPL(object):
    def __init__(self):
        self.__feature_col = ['speed', 'acceleration', 'accelerationX', 'accelerationY', 'accelerationZ']
        self.__label_col = 'flag'
        self.__time_step = 10
        self.__out_num = 1
        self.__hide_num = [15]
        self.__batch_size = 13
        self.__lstm_mode = LSTM(len(self.__feature_col), self.__time_step, self.__out_num, self.__hide_num,self.__batch_size)
        return

    def excute(self, path, train=True):
        data_parse = data_frame(path, self.__batch_size, self.__time_step, self.__feature_col, self.__label_col)
        log_configure.init_log(log_name='lstm', log_path='./lstm-log')
        optimize = gradient_descent(train_step=200, mode_path='./model/')
        if train:
            self.__lstm_mode.train(data_parse.next_batch, optimize.generalization_optimize)
        else:
            self.__lstm_mode.predict(data_parse.next_test, optimize.generalization_predict)

        return
