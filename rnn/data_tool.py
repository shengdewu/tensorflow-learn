import pandas as pd
import numpy as np
import logging
import os
import re

class data_frame(object):
    def __init__(self, path, batch_size, time_step, feature_column, label_colum, time_step_column, fileter_key):
        self.__batch_size = batch_size
        self.__time_step = time_step
        self.__next_batch = 0
        self.__feature_column = feature_column
        self.__label_colum = label_colum
        self.__path = path
        self.__index = 0
        self.__time_step_column = time_step_column
        self.__filter_key = fileter_key
        self.__max_step = 0
        self.__data = self._convert_list()
        return

    def get_train_step(self):
        return self.__max_step

    def _seek_data_source_file(self,data_source_path, filter_key, pos=1):
        '''
        :param data_source_path:  数据源路径
        :param filter_key: 数据源包含字符串
        :param pos: 匹配位置 0 开始，1，任意
        :return: 文件列表
        '''

        pattern = re.compile(filter_key)

        data_source_list = os.listdir(data_source_path)
        data_sourc = []
        for data_source_name in data_source_list:
            source_name = data_source_path + '/' + data_source_name
            if not os.path.isfile(source_name):
                continue
            if pos == 0:
                if pattern.match(data_source_name) is None:
                    continue
            else:
                if pattern.search(data_source_name) is None:
                    continue
            data_sourc.append(source_name)
        return data_sourc

    def _append_data(self, data, data_array, label_array):
        label = data.loc[:, self.__label_colum]
        label.replace(-2, 0, inplace=True)
        vmin = data.loc[:, self.__feature_column].min(axis=0)
        vmax = data.loc[:, self.__feature_column].max(axis=0)
        data = (data.loc[:, self.__feature_column]-vmin) / (vmax - vmin)
        data_array.append(data.to_numpy())
        label_array.append(label.to_numpy().reshape(self.__time_step, 1))
        return

    def _convert_list(self):
        use_col = self.__feature_column.copy()
        use_col.append(self.__label_colum)
        use_col.append(self.__time_step_column)
        data_array = []
        label_array = []
        df = pd.read_csv(self.__path, engine='c', usecols=use_col, encoding='gbk')
        df = df.loc[(df[self.__label_colum] == 1) | (df[self.__label_colum] == 0)]
        for dg in df.groupby(by=self.__time_step_column):
            datag = dg[1].drop(columns=[self.__time_step_column])
            if self.__time_step != datag.shape[0]:
                logging.warning('time step not equal{}-{}'.format(self.__time_step, datag.shape[0]))
                datag.index = [x for x in range(datag.shape[0])]
                for index in range(0, datag.shape[0], self.__time_step):
                    d = datag.iloc[index: index+self.__time_step]
                    self._append_data(d, data_array, label_array)
            else:
                self._append_data(datag, data_array, label_array)
        self.__max_step = len(data_array) // self.__batch_size
        print('total : {}-{}'.format(len(data_array), self.__max_step))
        return data_array, label_array

    def next_batch(self):
        if self.__next_batch + self.__batch_size > len(self.__data[0]):
            self.__next_batch = 0
            print('over data')
        data = np.array(self.__data[0][self.__next_batch:self.__next_batch+self.__batch_size])
        label = np.array(self.__data[1][self.__next_batch:self.__next_batch + self.__batch_size])
        self.__next_batch += self.__batch_size
        return data, label

    def next_test(self):
        test_data = None
        if self.__index < len(self.__data[0]):
            data = np.array(self.__data[0][self.__index:self.__index+1])
            label = np.array(self.__data[1][self.__index:self.__index+1])
            test_data = (data, label)
        self.__index += 1
        return test_data

