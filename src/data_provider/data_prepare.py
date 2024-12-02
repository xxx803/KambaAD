import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.dictionary_dot import DictionaryDot


class DataPrepare(object):
    def __init__(self, args):
        self._base_path = str(os.path.join(args.processed_path, args.dataset))
        self._dataset = args.dataset
        self._file_name = args.file_name

    def get(self):
        file_path = os.path.join(self._base_path, self._file_name)
        train_data = np.load(file_path + 'train.npy').astype(np.float32)
        test_data = np.load(file_path + 'test.npy').astype(np.float32)
        test_labels = np.load(file_path + 'labels.npy').astype(np.float32)
        data_src = DictionaryDot({
            'train': train_data,
            'test': test_data,
            'labels': test_labels,
        }).to_dot()
        # print(data_src.train.shape,data_src.test.shape,test_labels.shape)
        return data_src
