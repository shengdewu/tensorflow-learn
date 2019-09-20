import pandas as pd
import numpy as np

class data_frame(object):
    def __init__(self, path, batch_size, time_step, feature_column, label_colum):
        self.__batch_size = batch_size
        self.__time_step = time_step
        self.__next_batch = 0
        self.__feature_column = feature_column
        self.__label_colum = label_colum
        self.__path = path
        self.__data = self._convert_list()
        self.__index = 0
        return

    def _convert_list(self):
        use_col = self.__feature_column.copy()
        use_col.append(self.__label_colum)
        data_frame = pd.read_csv(self.__path, engine='c', usecols=use_col, encoding='gbk')
        label = data_frame.loc[:,self.__label_colum]
        median = data_frame.loc[:,self.__feature_column].median(axis=0)
        data_frame_tmp = data_frame.loc[:,self.__feature_column] / median
        del data_frame
        data_frame = data_frame_tmp
        data_frame.insert(data_frame.shape[1], column=self.__label_colum, value=label)
        data_array = []
        label_array = []
        for i in range(0,data_frame.shape[0], self.__time_step):
            data = data_frame.iloc[i:i+self.__time_step].loc[:, self.__feature_column].to_numpy().tolist()
            label = data_frame.iloc[i:i + self.__time_step].loc[:, self.__label_colum].to_numpy().reshape(self.__time_step, 1).tolist()
            data_array.append(data)
            label_array.append(label)
        return data_array, label_array

    def next_batch(self):
        if self.__next_batch + self.__batch_size < len(self.__data[0]):
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

