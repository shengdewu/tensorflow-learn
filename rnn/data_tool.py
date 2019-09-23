import pandas as pd
import numpy as np
import logging
class data_frame(object):
    def __init__(self, path, batch_size, time_step, feature_column, label_colum, time_step_column):
        self.__batch_size = batch_size
        self.__time_step = time_step
        self.__next_batch = 0
        self.__feature_column = feature_column
        self.__label_colum = label_colum
        self.__path = path
        self.__index = 0
        self.__time_step_column = time_step_column
        self.__data = self._convert_list()
        return

    def _append_data(self, data, data_array, label_array):
        label = data.loc[:, self.__label_colum]
        median = data.loc[:, self.__feature_column].median(axis=0)
        data = data.loc[:, self.__feature_column] / (median + 1)
        data_array.append(data.to_numpy())
        label_array.append(label.to_numpy().reshape(self.__time_step, 1))
        return

    def _convert_list(self):
        use_col = self.__feature_column.copy()
        use_col.append(self.__label_colum)
        use_col.append(self.__time_step_column)
        df = pd.read_csv(self.__path, engine='c', usecols=use_col, encoding='gbk')
        data_array = []
        label_array = []
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

        return data_array, label_array

    def next_batch(self):
        if self.__next_batch + self.__batch_size > len(self.__data[0]):
            self.__next_batch = 0
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

