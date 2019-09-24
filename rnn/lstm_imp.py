from rnn.data_tool import data_frame
from rnn.lstm import LSTM
from optimizer.gradient_descent import gradient_descent
from log.log_configure import log_configure
import logging
import pandas as pd

class LSTM_IMPL(object):
    def __init__(self,
                 log_path,
                 file_key,
                 batch_size=13,
                 time_step=22,
                 out_num=1,
                 hide_num=(15, 10),
                 feature_col=('speed', 'acceleration', 'accelerationX', 'accelerationY', 'accelerationZ'),
                 label_col='flag',
                 time_step_column='flagtime',
                 mode_path='./mode/checkout'):
        log_configure.init_log(log_name='lstm', log_path=log_path)
        self.__feature_col = list(feature_col)#['speed', 'acceleration', 'accelerationX', 'accelerationY', 'accelerationZ']
        self.__label_col = label_col#'flag'
        self.__time_step = time_step#22
        self.__out_num = out_num#1
        self.__hide_num = hide_num#[15]
        self.__batch_size = batch_size#1
        self.__time_step_column = time_step_column#'flagtime'
        self.__file_key = file_key#'lstm-[0-9]'
        self.__mode_path = mode_path
        logging.info('start init lstm use feature_col {}, label_col {}, time_step {}, out_num {}, hide_num {}, batch_size {}, time_step_column {}, file_key {}'.format(feature_col, label_col, time_step, out_num, hide_num, batch_size, time_step_column, file_key))
        self.__lstm_mode = LSTM(len(self.__feature_col), self.__time_step, self.__out_num, self.__hide_num, self.__batch_size)
        return

    def excute(self, path, train=True):
        data_parse = data_frame(path, self.__time_step, self.__feature_col, self.__label_col, self.__time_step_column, self.__file_key)
        optimize = gradient_descent(mode_path=self.__mode_path, batch_size=self.__batch_size, save_freq=1)
        if train:
            logging.debug('start train...')
            self.__lstm_mode.train(data_parse.next_batch, optimize.generalization_optimize)
        else:
            logging.debug('start test...')
            predict = self.__lstm_mode.predict(data_parse.next_batch, optimize.generalization_predict)
            col = list()
            col.append(self.__label_col)
            col.append('predict')
            predict_frame = pd.DataFrame(data=predict, columns=col)
            tp = predict_frame.loc[(predict_frame['predict']==1) & (predict_frame[self.__label_col] == 1)]
            fn = predict_frame.loc[(predict_frame['predict']==0) & (predict_frame[self.__label_col] == 1)]
            fp = predict_frame.loc[(predict_frame['predict']==1) & (predict_frame[self.__label_col] == 0)]
            tp = tp.shape[0]
            fn = fn.shape[0]
            fp = fp.shape[0]

            print('recall=tp/(tp+fn):{},precesion=tp/(tp+fp):{}'.format(tp/(tp+fn), tp/(tp+fp)))
            predict_frame.to_csv(path + '/predict.csv', index=False)
        return
