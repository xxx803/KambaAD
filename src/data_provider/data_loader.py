from torch.utils.data import DataLoader

from src.data_provider.data_prepare import DataPrepare
from src.data_provider.data_set import PointWindowDataset
from utils.dictionary_dot import DictionaryDot


class SlidDataloader:
    def __init__(self, args):
        self._batch_size = args.batch_size
        self._window_size = args.window_size
        self._data = DataPrepare(args=args).get()

    def get(self):
        """
        构建数据加载器
        :return:
        """
        train_set = PointWindowDataset(data=self._data.train, window_size=self._window_size)
        test_set = PointWindowDataset(data=self._data.test, window_size=self._window_size)

        train_loader = DataLoader(dataset=train_set, batch_size=self._batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(dataset=test_set, batch_size=self._batch_size, shuffle=False, drop_last=False)

        loader = DictionaryDot({
            'train': {
                'data': self._data.train,
                'loader': train_loader,
                'features': self._data.train.shape[1]
            },
            'test': {
                'data': self._data.test,
                'loader': test_loader,
                'labels': self._data.labels,
            },
            'valid': {
                'data': self._data.test,
                'loader': test_loader,
                'labels': self._data.labels,
            },
        }).to_dot()

        return loader
