import pandas as pd
import numpy as np
import logging
import os
import re
import random

class data_frame(object):
    def __init__(self, path, time_step, feature_column, label_colum, time_step_column, file_key):
        self.__time_step = time_step
        self.__feature_column = feature_column
        self.__label_colum = label_colum
        self.__path = path
        self.__index = 0
        self.__time_step_column = time_step_column
        self.__next_batch = 0
        self.__file_key = file_key
        self.__position = 0
        self.__test, self.__train = self._featch_data()
        return

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
            print(source_name)
            data_sourc.append(source_name)
        return data_sourc

    def _append_data(self, data, data_array, label_array):
        label = data.loc[:, self.__label_colum]
        label.replace(-2, 0, inplace=True)
        l = label.iloc[0]
        if l == 1:
            self.__position += 1
        larray = np.zeros(shape=(2,), dtype=np.int)
        larray[l] = 1
        vmin = data.loc[:, self.__feature_column].min(axis=0)
        vmax = data.loc[:, self.__feature_column].max(axis=0)

        normal_data = data.loc[:, self.__feature_column].copy()
        denm = vmax - vmin
        is_zero = denm.loc[denm.isin([0])]
        if is_zero.empty:
            normal_data = (normal_data - vmin) / denm
        else:
            normal_data_dict = dict()
            zero_column = list(is_zero.index)
            for column in self.__feature_column:
                tmp_data = data.loc[:, column]
                tmp_max = tmp_data.max()
                tmp_min = tmp_data.min()
                tmp_normal_data = tmp_data
                if column in zero_column:
                    if tmp_max != 0 and tmp_max != 1:
                        tmp_normal_data = tmp_data/tmp_max
                else:
                    tmp_normal_data = (tmp_data-tmp_min)/(tmp_max-tmp_min)
                normal_data_dict[column] = tmp_normal_data
            normal_data = pd.DataFrame(normal_data_dict)

        data_array.append(normal_data.to_numpy())
        label_array.append(larray)
        return

    def _featch_data(self):
        use_col = self.__feature_column.copy()
        use_col.append(self.__label_colum)
        use_col.append(self.__time_step_column)

        file_list = self._seek_data_source_file(self.__path, self.__file_key, 1)

        data = self._convert_list(use_col, file_list)
        index = [x for x in range(len(data[0]))]
        k = int(len(data[0]) * 0.25)
        test_index = random.sample(index, k)
        print('featch test {}'.format(test_index))
        test_data_x = [data[0][x] for x in test_index]
        test_data_y = [data[1][x] for x in test_index]

        train_index = set(index).difference(set(test_index))
        print('featch train {}'.format(train_index))
        train_data_x = [data[0][x] for x in train_index]
        train_data_y = [data[1][x] for x in train_index]
        return (test_data_x, test_data_y), (train_data_x, train_data_y)

    def _convert_list(self, use_col, file_list):
        data_array = []
        label_array = []
        for fl in file_list:
            df = pd.read_csv(fl, engine='c', usecols=use_col, encoding='gbk')
            df = df.loc[(df[self.__label_colum] == 1) | (df[self.__label_colum] == 0)]
            for dg in df.groupby(by=self.__time_step_column):
                if self.__time_step != dg[1].shape[0]:
                    #print('file{}-key{}:time step {} not match actual {}'.format(fl, dg[0], self.__time_step, dg[1].shape[0]))
                    dg[1].index = [x for x in range(dg[1].shape[0])]
                    for index in range(0, dg[1].shape[0], self.__time_step):
                        d = dg[1].iloc[index: index+self.__time_step]
                        count = d.duplicated(subset=self.__time_step_column, keep=False).value_counts()[True]
                        label = d.duplicated(subset=self.__label_colum, keep=False).value_counts()[True]
                        if self.__time_step != count or self.__time_step != label:
                            raise RuntimeError('step is not equal {}!= {} or {}'.format(self.__time_step, label, count))
                        sample = d.drop(columns=[self.__time_step_column])
                        self._append_data(sample, data_array, label_array)
                else:
                    label = dg[1].duplicated(subset=self.__label_colum, keep=False).value_counts()[True]
                    if self.__time_step != label:
                        raise RuntimeError('step is not equal label not match {}!= {}'.format(self.__time_step, label))
                    sample = dg[1].drop(columns=[self.__time_step_column])
                    self._append_data(sample, data_array, label_array)
        print('total : {}/{}'.format(len(data_array), self.__position))
        return data_array, label_array

    def next_batch(self, batch_size, train=True):
        data = None
        label = None
        if train:
            if self.__next_batch + batch_size <= len(self.__train[0]):
                data = np.array(self.__train[0][self.__next_batch:self.__next_batch + batch_size])
                label = np.array(self.__train[1][self.__next_batch:self.__next_batch + batch_size])
            self.__next_batch += batch_size
        else:
            if batch_size is None:
                data = np.array(self.__test[0])
                label = np.array(self.__test[1])
            else:
                if self.__next_batch + batch_size <= len(self.__test[0]):
                    data = np.array(self.__test[0][self.__next_batch:self.__next_batch + batch_size])
                    label = np.array(self.__test[1][self.__next_batch:self.__next_batch + batch_size])
                self.__next_batch += batch_size
        return data, label

    def clean(self):
        self.__next_batch = 0
        return



