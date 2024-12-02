import torch
from torch.utils.data import TensorDataset, Dataset


class SlideDataset(object):
    def __init__(self, data, window_size):
        self._data = data
        self._window_size = window_size

    @staticmethod
    def convert_to_windows(data, window_size):
        data = torch.FloatTensor(data)
        data_x = []
        data_y = []
        for i, _ in enumerate(data[:-window_size, :]):
            data_x.append(data[i:i + window_size, :])
            data_y.append(data[i + window_size - 1:i + window_size, :])
        data_x = torch.stack(data_x)
        data_y = torch.stack(data_y)
        return data_x, data_y

    def get(self):
        x, y = self.convert_to_windows(self._data, self._window_size)
        data_set = TensorDataset(x, y)
        return data_set


class PointWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __getitem__(self, index):
        x = self.data[index: index + self.window_size]
        y = self.data[index + self.window_size - 1: index + self.window_size]
        return x, y

    def __len__(self):
        return len(self.data[:-self.window_size, :])
