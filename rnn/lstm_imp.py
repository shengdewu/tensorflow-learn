from rnn.data_tool import data_frame
from rnn.lstm import LSTM
from optimizer.gradient_descent import gradient_descent
from log.log_configure import log_configure
import logging
import pandas as pd
from tool.parse_config import parse_config

class LSTM_IMPL(object):
    def __init__(self,
                 log_path,
                 train_file_key,
                 test_file_key,
                 feature_col=('speed', 'acceleration', 'accelerationX', 'accelerationY', 'accelerationZ'),
                 label_col='flag',
                 time_step_column='flagtime',
                 config_path=''):
        log_configure.init_log(log_name='lstm', log_path=log_path)
        self.__feature_col = list(feature_col)#['speed', 'acceleration', 'accelerationX', 'accelerationY', 'accelerationZ']
        self.__label_col = label_col#'flag'
        self.__time_step_column = time_step_column#'flagtime'
        self.__train_file_key = train_file_key#'lstm-[0-9]'
        self.__test_file_key = test_file_key

        self.__config = parse_config.get_config(config_path)
        logging.info('mode param: feature_col {}, label_col {}, time_step_column {}, config {}'.format(feature_col, label_col, time_step_column, self.__config))
        self.__lstm_mode = LSTM(self.__config['input_num'],
                                self.__config['time_step'],
                                self.__config['output_num'],
                                tuple(self.__config['cell_unit']),
                                self.__config['batch_size'])

        self.__optimize = gradient_descent(self.__config['learn_rate'],
                                           self.__config['decay_rate'],
                                           self.__config['moving_decay'],
                                           self.__config['regularize_rate'],
                                           self.__config['max_iter_times'],
                                           self.__config['mode_path'],
                                           self.__config['update_mode_freq'],
                                           self.__config['batch_size'])
        return

    def excute(self, path):

        logging.debug('start train...')
        train_data_parse = data_frame(path, self.__config['time_step'], self.__feature_col, self.__label_col,self.__time_step_column, self.__train_file_key)
        self.__lstm_mode.train(train_data_parse.next_batch, self.__optimize.generalization_optimize)
        train_data_parse.clean()

        logging.debug('start test...')
        test_data_parse = data_frame(path, self.__config['time_step'], self.__feature_col, self.__label_col,self.__time_step_column, self.__test_file_key)
        predict = self.__lstm_mode.predict(test_data_parse.next_batch, self.__optimize.generalization_predict)
        test_data_parse.clean()
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
