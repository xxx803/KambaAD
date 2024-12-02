import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.supports.affiliation.generics import convert_vector_to_events
from src.supports.affiliation.metrics import pr_from_events
from src.supports.spot import SPOT
from utils.dictionary_dot import DictionaryDot
from src.metrics.metrics import combine_all_evaluation_scores
from src.metrics.f1_score_f1_pa import get_adjust_F1PA
from src.metrics.vus.metrics import get_range_vus_roc


# import warnings
#
# warnings.filterwarnings("ignore")


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f_score = 2 * precision * recall / (precision + recall + 0.00001)
    return f_score


class BestEvaluator:
    def __init__(self, train_scores, valid_scores, test_labels):
        self.train_scores = train_scores
        self.valid_scores = valid_scores
        self.test_labels = test_labels

    @staticmethod
    def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
        """
        Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
        Args:
                score (np.ndarray): The anomaly score
                label (np.ndarray): The ground-truth label
                threshold (float): The threshold of anomaly score.
                        A point is labeled as "anomaly" if its score is lower than the threshold.
                pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
                calc_latency (bool):
        Returns:
                np.ndarray: predict labels

        Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
        """
        if label is None:
            predict = score > threshold
            return predict, None

        if pred is None:
            if len(score) != len(label):
                raise ValueError("score and label must have the same length")
            predict = score > threshold
        else:
            predict = pred

        actual = np.array((label > 0.1), dtype=bool)

        anomaly_state = False
        anomaly_count = 0
        latency = 0

        for i in range(len(predict)):
            if any(actual[max(i, 0): i + 1]) and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
        if calc_latency:
            return predict, latency / (anomaly_count + 1e-4)
        else:
            return predict

    @staticmethod
    def get_threshold(init_score, test_score, q=1e-2):
        s = SPOT(q=q)
        s.fit(init_score, test_score)
        s.initialize(verbose=False)
        ret = s.run()
        threshold = np.mean(ret['thresholds'])

        return threshold

    @staticmethod
    def get_affiliation_metrics(label, pred):
        events_pred = convert_vector_to_events(pred)
        events_label = convert_vector_to_events(label)
        Trange = (0, len(pred))

        result = pr_from_events(events_pred, events_label, Trange)
        P = result['precision']
        R = result['recall']
        F = 2 * P * R / (P + R)

        return P, R, F

    def calc_seq(self, score, label, threshold):
        predict = self.adjust_predicts(score, label, threshold)
        return calc_point2point(predict, label)

    def get_results(self, threshold):
        predict = (self.valid_scores > threshold).astype(int)
        combine_all_evaluation_scores(self.test_labels, predict)

    def evaluate1(self):
        """
        Find the best-f1 score by searching best `threshold` in [`start`, `end`).
        Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
        """
        start, end, step_num = np.percentile(self.valid_scores, 80), np.percentile(self.valid_scores, 99.99), 1000
        print(start, end, step_num)
        search_step, search_range, search_lower_bound = step_num, end - start, start
        threshold = search_lower_bound
        best_result = DictionaryDot(args_obj={
            'f_score': -1.0,
            'threshold': threshold,
        }).to_dot()
        for i in range(search_step):
            threshold += search_range / float(search_step)
            f_score = self.calc_seq(self.valid_scores, self.test_labels, threshold)
            if f_score > best_result.f_score:
                best_result.f_score = f_score
                best_result.threshold = threshold
        best_result.scores = self.valid_scores
        best_result.labels = self.test_labels
        print(f'f_score:{best_result.f_score * 100:.2f},threshold:{best_result.threshold}')
        return best_result

    def combine_indicator(self, threshold):
        predictions = self.valid_scores > threshold
        res = combine_all_evaluation_scores(predictions, self.test_labels)
        best_result = (
            f"\t{'Accuracy':<22}:  {res['pa_accuracy']:.4f}\n"
            f"\t{'Precision':<22}:  {res['pa_precision']:.4f}\n"
            f"\t{'Recall':<22}:  {res['pa_recall']:.4f}\n"
            f"\t{'F-score':<22}:  {res['pa_f_score']:.4f}\n"
            f"\t{'MCC_score':<22}:  {res['MCC_score']:.4f}\n"
            f"\t{'Affiliation precision':<22}:  {res['Affiliation precision']:.4f}\n"
            f"\t{'Affiliation recall':<22}:  {res['Affiliation recall']:.4f}\n"
            f"\t{'R_AUC_ROC':<22}:  {res['R_AUC_ROC']:.4f}\n"
            f"\t{'R_AUC_PR':<22}:  {res['R_AUC_PR']:.4f}\n"
            f"\t{'VUS_ROC':<22}:  {res['VUS_ROC']:.4f}\n"
            f"\t{'VUS_PR':<22}:  {res['VUS_PR']:.4f}\n"
        )
        return best_result

    def evaluate2(self, q=1e-2):
        threshold = self.get_threshold(self.train_scores, self.valid_scores, q=q)
        return self.combine_indicator(threshold)

    def evaluate3(self, anomaly_ratio):
        combined_energy = np.concatenate([self.train_scores, self.valid_scores], axis=0)
        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
        return self.combine_indicator(threshold)
