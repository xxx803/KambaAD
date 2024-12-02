import os
import pickle
import time
import argparse
import numpy as np
import torch

from src.experiments.experiments_basic import ExperimentsBasic
from src.supports.early_stopping import EarlyStopping, adjust_learning_rate
from utils.out_console import out_console, Color, Style
from src.supports.best_evaluator import BestEvaluator
from src.metrics.metrics import combine_all_evaluation_scores


class ExperimentsMain(ExperimentsBasic):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        super(ExperimentsMain, self).__init__(args=args)

    def train(self, itr):
        f_name = '' if len(self.args.file_name.strip()) == 0 else self.args.file_name.strip()[:-1]
        title = f'\n{itr + 1:04d} - Model Name:{self.args.model_name}, Dataset Name:{self.args.dataset}/{f_name}'
        out_console.out_title(title, color=Color.GREEN.value, style=Style.DEFAULT.value)

        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        train_steps = len(self.data_info.train.loader)

        for epoch in range(self.args.epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()

            for i, (x, y) in enumerate(self.data_info.train.loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                iter_count += 1

                _, window = self.model(x)
                if y.ndim == 3:
                    y = y.squeeze(1)

                # point_loss = torch.sqrt(self.point_criterion(y, point))
                window_loss = torch.sqrt(self.window_criterion(x, window))
                loss = window_loss

                # if (i + 1) % 100 == 0:
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                #     print('\tspeed: {*100:.2f}s/iter; left time: {*100:.2f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                loss.backward()
                self.optimizer.step()

            valid_loss = self.valid(self.data_info.test.loader)
            # print(
            #     "Epoch: {0}, Cost time: {1:.3f}s ".format(
            #         epoch + 1, time.time() - epoch_time))
            early_stopping(valid_loss, self.model, self.checkpoint_name)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.model.lr)

    def valid(self, valid_loader):
        self.model.eval()
        loss_lst = []
        for x, y in self.data_info.train.loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            _, window = self.model(x)
            if y.ndim == 3:
                y = y.squeeze(1)

            # point_loss = torch.sqrt(self.point_criterion(y, point))
            window_loss = torch.sqrt(self.window_criterion(x, window))
            loss = window_loss
            loss_lst.append(loss.detach().cpu().numpy().item())
        losses = np.array(loss_lst)

        return np.average(losses)

    def test(self):
        # self.model.load_state_dict(torch.load(self.checkpoint_name))
        self.model.eval()
        valid_anomaly_scores = self._get_anomaly_scores(self.data_info.valid.loader, self.data_info.valid.data)
        train_anomaly_scores = self._get_anomaly_scores(self.data_info.train.loader, self.data_info.train.data)
        labels = self.data_info.valid.labels[self.args.window_size:]
        best_evaluator = BestEvaluator(train_anomaly_scores, valid_anomaly_scores, labels)
        if self.args.threshold == 'threshold1':
            best_result = best_evaluator.evaluate1()
        elif self.args.threshold == 'threshold2':
            best_result = best_evaluator.evaluate2(q=self.args.q)
        elif self.args.threshold == 'threshold3':
            best_result = best_evaluator.evaluate3(self.args.anomaly_ratio)
        else:
            raise ValueError('Invalid threshold')
        return best_result

    def print_result(self):
        print('-------------------------', self.result_name)
        if self.args.dataset in ['MSL', 'SMAP', 'SMD']:
            dir_name = os.path.dirname(self.result_name)
            file_list = os.listdir(dir_name)
            predict_lst, actual_lst = [], []
            idx = 0
            for file_name in file_list:
                if file_name.startswith(self.args.dataset):
                    predictions, labels = self.get_result(os.path.join(dir_name, file_name))
                    predict_lst.append(predictions)
                    actual_lst.append(labels)
                    idx += 1
                    title = f'{idx:04d} {str(os.path.join(dir_name, file_name))}'
                    out_console.out_title(title, color=Color.CYAN.value, style=Style.ITALIC.value, level=0)
                    self.print_indicator(predictions, labels)
            predicts = np.concatenate(predict_lst)
            actuals = np.concatenate(actual_lst)
            out_console.out_line('Merge best f_score', mark='**', color=Color.CYAN.value, style=Style.ITALIC.value, length=80)
            self.print_indicator(predicts, actuals)
        elif 'NIPS' in self.args.dataset:
            dir_name = os.path.dirname(self.result_name)
            file_list = os.listdir(dir_name)
            idx = 0
            for file_name in file_list:
                if file_name.startswith('NIPS'):
                    predictions, labels = self.get_result(os.path.join(dir_name, file_name))
                    idx += 1
                    title = f'{idx:04d} {str(os.path.join(dir_name, file_name))}'
                    out_console.out_title(title, color=Color.CYAN.value, style=Style.ITALIC.value, level=0)
                    self.print_indicator(predictions, labels)
        else:
            title = f'{self.result_name}'
            predictions, labels = self.get_result(self.result_name)
            out_console.out_title(title, color=Color.CYAN.value, style=Style.ITALIC.value, level=0)
            self.print_indicator(predictions, labels)
