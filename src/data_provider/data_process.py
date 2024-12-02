import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.out_console import out_console, Color, Style
from src.arguments import build_args


class DataPreprocessor(object):
    def __init__(self, args, dataset):
        dataset_path = 'SMAP_MSL' if dataset in ['SMAP', 'MSL'] else dataset
        self._source_path = str(os.path.join('../.' + args.source_path, dataset_path))
        self._dataset = dataset
        self._target_path = self._create_target_path('../.' + args.processed_path, dataset)

    @staticmethod
    def _create_target_path(target_path, dataset):
        target_path = str(os.path.join(target_path, dataset))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        return target_path

    def _export_ucr(self):
        file_list = os.listdir(self._source_path)
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            d_num, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(self._source_path, filename), dtype=np.float64, delimiter=',')
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros((test.shape[0],), dtype=np.int32)
            labels[vals[1] - vals[0]:vals[2] - vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1)

            scaler = MinMaxScaler()
            scaler.fit(train)
            train, test = scaler.transform(train), scaler.transform(test)

            self.save_files(train, test, labels, f_name=str(d_num))

    def _export_swat(self):
        file = os.path.join(self._source_path, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]

        # train, min_a, max_a = self.normalize2(df_train.values)
        # test, _, _ = self.normalize2(df_test.values, min_a, max_a)
        xscaler = MinMaxScaler()
        xscaler.fit(df_train.values)
        train, test = xscaler.transform(df_train.values), xscaler.transform(df_test.values)

        labels = (pd.read_json(file, lines=True)[['noti']][7000:12000] + 0).values.reshape(-1)

        scaler = MinMaxScaler()
        scaler.fit(train)
        train, test = scaler.transform(train), scaler.transform(test)

        self.save_files(train, test, labels)

    def _export_nab(self):
        file_list = os.listdir(self._source_path)
        with open(self._source_path + '/labels.json') as f:
            label_dct = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'):
                continue
            df = pd.read_csv(self._source_path + '/' + filename)
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in label_dct['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp) + 0) == 1)[0][0]
                labels[index - 4:index + 4] = 1

            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1)

            xscaler = MinMaxScaler()
            xscaler.fit(train)
            train, test = xscaler.transform(train), xscaler.transform(test)

            self.save_files(train, test, labels, f_name=filename.replace('.csv', ''))

    def _export_mba(self):
        ls = pd.read_excel(os.path.join(self._source_path, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(self._source_path, 'train.xlsx'))
        test = pd.read_excel(os.path.join(self._source_path, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros((test.shape[0]), dtype=np.int32)

        for i in range(-20, 20):
            labels[ls + i] = 1

        self.save_files(train, test, labels)

    def _export_smap_msl(self):
        file = os.path.join(self._source_path, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == self._dataset]
        filenames = values['chan_id'].values.tolist()
        for file_name in filenames:
            train = np.load(f'{self._source_path}/train/{file_name}.npy')
            test = np.load(f'{self._source_path}/test/{file_name}.npy')

            labels = np.zeros((test.shape[0],), dtype=np.int32)
            indices = values[values['chan_id'] == file_name]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1]] = 1

            self.save_files(train, test, labels, f_name=file_name)

    def _export_smd(self):
        src_path = os.path.join(self._source_path, 'train')
        file_list = os.listdir(src_path)
        for file_name in file_list:
            if file_name.endswith('.txt'):
                train = np.genfromtxt(fname=os.path.join(self._source_path, 'train', file_name), delimiter=',', dtype=np.float32)
                test = np.genfromtxt(fname=os.path.join(self._source_path, 'test', file_name), delimiter=',', dtype=np.float32)
                labels = np.genfromtxt(fname=os.path.join(self._source_path, 'labels', file_name), delimiter=',', dtype=np.float32)

                # scaler = MinMaxScaler()
                # scaler.fit(train)
                # train, test = scaler.transform(train), scaler.transform(test)

                self.save_files(train=train, test=test, labels=labels, f_name=file_name.strip(".txt"))

    def _export_synthetic(self):
        train_file = os.path.join(self._source_path, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(self._source_path, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = dat.values[:, :split].reshape(split, -1)
        test = dat.values[:, split:].reshape(split, -1)

        scaler = MinMaxScaler()
        scaler.fit(train)
        train, test = scaler.transform(train), scaler.transform(test)

        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros((test.shape[0],), dtype=np.int32)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30] = 1

        test += labels.reshape(-1, 1) * np.random.normal(0.75, 0.1, test.shape)

        self.save_files(train, test, labels)

    def get_wadi_file(self, file_name):
        hand_out = {
            'train': 'WADI_14days.csv',
            'test': 'WADI_attackdata.csv',
            'train_new': 'WADI_14days_new.csv',
            'test_new': 'WADI_attackdataLABLE.csv'
        }
        res = os.path.join(self._source_path, hand_out[file_name])
        return res

    @staticmethod
    def recover_date(str1, str2):
        return str1 + " " + str2

    def build_wadi_data(self, x, x_new):
        """
        1 更新第2版权时间格式
            第2版中的数据时间格式存在错误，因为两个版本都存在 Row 这行，所以利用 Row 将时间格式替换成第一个版本中的时间数据即可。
        :param x: 第1版训练集或测试集
        :param x_new: 第2版训练集或测试集
        """
        x['Time'] = x['Time'].apply(lambda p: datetime.strptime(p, '%I:%M:%S.000 %p').strftime('%I:%M:%S'))
        x["datetime"] = x.apply(lambda p: self.recover_date(p['Date'], p['Time']), axis=1)
        x["datetime"] = pd.to_datetime(x['datetime'])
        x = x[['Row', 'datetime']]
        x = pd.merge(x_new, x, how='left', on='Row')
        del x['Row']
        del x['Date']
        del x['Time']
        del x['datetime']
        return x

    def _export_wadi(self):
        train_new = pd.read_csv(self.get_wadi_file('train_new'))
        test_new = pd.read_csv(self.get_wadi_file('test_new'), skiprows=1)
        train = pd.read_csv(self.get_wadi_file('train'), skiprows=4)
        test = pd.read_csv(self.get_wadi_file('test'))

        # 去掉第1版字段前的路径
        test_columns = test.columns.tolist()
        for column in test_columns:
            test = test.rename(columns={column: column.split('\\')[-1]})

        # 去掉第2版字段中的空格
        test_new = test_new.rename(columns={'Row ': 'Row'})
        test_new = test_new.rename(columns={'Date ': 'Date'})

        # 更改第2版的 label 标签和值
        test_new = test_new.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'label'})
        test_new.loc[test_new['label'] == 1, 'label'] = 0
        test_new.loc[test_new['label'] == -1, 'label'] = 1

        # 处理训练集和测试集的时间和均值和填充
        train = self.build_wadi_data(train, train_new)
        test = self.build_wadi_data(test, test_new)

        # 查找空值字段并填充0
        null_cols = train.isnull().any()[train.isnull().any()]
        null_cols = [x for x in null_cols.index]
        train[null_cols] = 0
        test[null_cols] = 0

        # 步长为10, 取均值
        step = 10
        train = train.rolling(step).mean()
        train = train[(train.index + 1) % step == 0]
        test = test.rolling(step).mean()
        test = test[(test.index + 1) % step == 0]

        # 前填充
        train = train.ffill()
        test = test.ffill()

        # 处理标签
        labels = test['label'].apply(lambda s: 1 if s > 0.5 else 0).values

        # 转为 Numpy
        train = train.iloc[:, 0:].values
        test = test.iloc[:, 0:-1].values

        # 归一化
        scaler = MinMaxScaler()
        train, test = scaler.fit_transform(train), scaler.fit_transform(test)

        self.save_files(train, test, labels)

    def _export_psm(self):
        df_train = pd.read_csv(os.path.join(self._source_path, 'train.csv'), index_col=0)
        df_test = pd.read_csv(os.path.join(self._source_path, 'test.csv'), index_col=0)
        df_labels = pd.read_csv(os.path.join(self._source_path, 'test_label.csv'), index_col=0)
        df_train.fillna(0, inplace=True)
        df_test.fillna(0, inplace=True)
        df_labels.fillna(0, inplace=True)
        train, test, labels = df_train.values, df_test.values, df_labels.values.reshape(-1)

        xscaler = MinMaxScaler()
        xscaler.fit(train)
        train, test = xscaler.transform(train), xscaler.transform(test)

        self.save_files(train, test, labels)

    def _export_nips_ts(self):
        train = np.load(os.path.join(self._source_path, 'train.npy'))
        test = np.load(os.path.join(self._source_path, 'test.npy'))
        labels = np.load(os.path.join(self._source_path, 'labels.npy'))
        # xscaler = MinMaxScaler()
        # xscaler.fit(train)
        # train, test = xscaler.transform(train), xscaler.transform(test)

        self.save_files(train, test, labels)

    def _export_rtdw_aiops(self):
        file_list = os.listdir(self._source_path)
        for file_name in file_list:
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(self._source_path, file_name))
                df = df.fillna(0).values.astype(np.float32)

                split_dot = int(len(df) * 0.5)
                train, test, labels = df[:split_dot, :-1], df[split_dot:, :-1], df[split_dot:, -1]
                scaler = MinMaxScaler()
                scaler.fit(train)
                train, test = scaler.transform(train), scaler.transform(test)
                self.save_files(train=train, test=test, labels=labels, f_name=file_name.strip(".csv"))

    def save_files(self, train=None, test=None, labels=None, f_name=''):
        for file_style in ['train', 'test', 'labels']:
            out_file = os.path.join(self._target_path, f'{f_name}{"" if len(f_name) == 0 else "_"}{file_style}.npy')
            np.save(out_file, eval(file_style))
            print(f'{out_file} {eval(file_style).shape}')

    def export(self):
        hand_out = {
            'UCR': self._export_ucr,
            'SWaT': self._export_swat,
            'NAB': self._export_nab,
            'MBA': self._export_mba,
            'SMAP': self._export_smap_msl,
            'MSL': self._export_smap_msl,
            'SMD': self._export_smd,
            'synthetic': self._export_synthetic,
            'WADI': self._export_wadi,
            'PSM': self._export_psm,
            'NIPS_TS_Water': self._export_nips_ts,
            'NIPS_TS_CCard': self._export_nips_ts,
            'NIPS_TS_Swan': self._export_nips_ts,
            'NIPS_TS_Syn_Mulvar': self._export_nips_ts,
            'RTDWAIOps': self._export_rtdw_aiops,

        }
        hand_out[self._dataset]()


# ['UCR', 'SWaT', 'NAB', 'MBA', 'SMAP', 'MSL', 'SMD', 'synthetic', 'WADI', 'PSM']
@out_console.repeat(mark='><', length=80)
def execute():
    args = build_args()

    for idx, ds in enumerate(['SMAP']):
        out_console.out_line(f'{format(idx + 1, "02d")} {ds} ', mark='--', color=Color.LIGHTBLUE_EX, style=Style.ITALIC, length=80)
        DataPreprocessor(
            args=args,
            dataset=ds  # 数据集名
        ).export()


if __name__ == '__main__':
    execute()
