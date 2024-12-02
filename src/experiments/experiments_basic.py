import os
import pickle
import argparse
import torch
from torch import nn, optim
import numpy as np

from src.data_provider.data_loader import SlidDataloader
from src.models import KambaAD
from src.metrics.metrics import combine_all_evaluation_scores


class ExperimentsBasic:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._acquire_device()
        # self.args.file_name = self.get_file_name()
        self.checkpoint_name, self.result_name = self.get_save_checkpoint_name()
        self.data_info = SlidDataloader(args=args).get()
        self.args.features = self.data_info.train.features
        self.args.enc_in = self.data_info.train.features
        self.target_dims = None
        self.model = self._build_model().to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {num_params:,} trainable parameters.")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr)
        self.point_criterion = nn.MSELoss()
        self.window_criterion = nn.MSELoss()

    @staticmethod
    def _acquire_device() -> torch.device:
        """
        :return: 返回模型运行设备
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    def _build_model(self) -> nn.Module:
        """
        生成模型
        :return:
        """
        hand_out_model = {
            'KambaAD': KambaAD,
        }
        return hand_out_model[self.args.model_name].Model(configs=self.args).float()

    def get_save_checkpoint_name(self):
        model_save_path = os.path.join(self.args.output, 'checkpoints')
        result_save_path = os.path.join(self.args.output, 'results')
        print(model_save_path,result_save_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        checkpoint_name = os.path.join(
            str(model_save_path), self.args.dataset + '_' + self.args.file_name.strip() + 'checkpoint.pth'
        )
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        result_name = os.path.join(
            str(result_save_path), self.args.dataset + '_' + self.args.file_name.strip() + '.pkl'
        )
        return checkpoint_name, result_name

    def get_file_name(self):
        if self.args.dataset == 'SMD':
            f_name = self.args.file_name if len(self.args.file_name.strip()) > 0 else ''
        elif self.args.dataset == 'UCR':
            f_name = self.args.file_name if len(self.args.file_name.strip()) > 0 else '136_'
        elif self.args.dataset == 'NAB':
            f_name = self.args.file_name if len(self.args.file_name.strip()) > 0 else 'ec2_request_latency_system_failure_'
        elif self.args.dataset == 'SMAP':
            f_name = self.args.file_name if len(self.args.file_name.strip()) > 0 else 'P-1_'
        # elif self.args.dataset == 'MSL':
        #     f_name = self.args.file_name if len(self.args.file_name.strip()) > 0 else 'C-1_'
        elif self.args.dataset == 'AIOps':
            f_name = self.args.file_name if len(self.args.file_name.strip()) > 0 else 'instance23_'
        return f_name

    def _get_anomaly_scores(self, loader, data):
        self.model.eval()
        window_lst = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                _, window = self.model(x)
                # point_lst.append(point.detach().cpu().numpy())
                window_lst.append(window[:, -1, :].detach().cpu().numpy())

        windows = np.concatenate(window_lst, axis=0)
        actual = data[self.args.window_size:]
        anomaly_scores = np.zeros_like(actual)
        for i in range(windows.shape[1]):
            current_score = self.args.gamma * np.sqrt((windows[:, i] - actual[:, i]) ** 2) * 1000
            anomaly_scores[:, i] = current_score
        anomaly_scores = np.mean(anomaly_scores, 1)
        return anomaly_scores

    @staticmethod
    def get_result(file_name):
        with open(os.path.join(file_name), 'rb') as f:
            result = pickle.load(f)
        predictions = result.scores > result.threshold
        print(f'{file_name}:{result.seed}')
        return predictions, result.labels

    @staticmethod
    def print_indicator(predictions, labels):
        res_info = combine_all_evaluation_scores(predictions, labels)
        best_result = (
            f"\t{'Accuracy':<22}:  {res_info['pa_accuracy'] * 100:.2f}\n"
            f"\t{'F-score':<22}:  {res_info['pa_f_score'] * 100:.2f}\n"
            f"\t{'Affiliation precision':<22}:  {res_info['Affiliation precision'] * 100:.2f}\n"
            f"\t{'Affiliation recall':<22}:  {res_info['Affiliation recall'] * 100:.2f}\n"
            f"\t{'R_AUC_ROC':<22}:  {res_info['R_AUC_ROC'] * 100:.2f}\n"
            f"\t{'R_AUC_PR':<22}:  {res_info['R_AUC_PR'] * 100:.2f}\n"
            f"\t{'VUS_ROC':<22}:  {res_info['VUS_ROC'] * 100:.2f}\n"
            f"\t{'VUS_PR':<22}:  {res_info['VUS_PR'] * 100:.2f}\n"
            f"\t{'Precision':<22}:  {res_info['pa_precision'] * 100:.2f}\n"
            f"\t{'Recall':<22}:  {res_info['pa_recall'] * 100:.2f}\n"
            # f"\t{'MCC_score':<22}:  {res_info['MCC_score']*100:.2f}\n"
        )
        print(best_result)

    def train(self, *args, **kwargs):
        pass

    def valid(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def print_result(self, *args, **kwargs):
        pass
