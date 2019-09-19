import pandas as pd

class data_frame(object):
    def __init__(self, path, batch_size, feature_column, label_colum):
        self.__batch_size = batch_size
        self.__next_batch = 0
        self.__data_frame = pd.read_csv(path, encoding='c')
        self.__total_row = self.__data_frame.shape[0]
        self.__feature_column = feature_column
        self.__label_colum = label_colum
        return

    def next_batch(self):
        batch_data = self.__data_frame.sample(self.__batch_size, random_state=1)
        return batch_data.loc[:, self.__feature_column], batch_data.loc[:, self.__label_colum]
