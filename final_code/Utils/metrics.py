import math
from typing import List, Callable, Union

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from scipy.stats import pearsonr


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def balanced_accuracy(
    targets: List[int], preds: List[float], threshold: float = 0.5
) -> float:
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return balanced_accuracy_score(targets, hard_preds)


def recall(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return recall_score(targets, hard_preds)


def precision(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return precision_score(targets, hard_preds)


def sensitivity(
    targets: List[int], preds: List[float], threshold: float = 0.5
) -> float:
    """
    Computes the sensitivity of a binary prediction task using a given threshold for generating hard predictions.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed sensitivity.
    """
    return recall(targets, preds, threshold)


def specificity(
    targets: List[int], preds: List[float], threshold: float = 0.5
) -> float:
    """
    Computes the specificity of a binary prediction task using a given threshold for generating hard predictions.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed specificity.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    cm = confusion_matrix(targets, hard_preds).ravel()
    try:
        tn, fp, fn, tp = cm
        return tn / float(tn + fp)
    except ValueError:
        return float('nan')


def mcc(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the Matthews correlation coefficient of a binary prediction task using a given threshold for generating hard predictions.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed Matthews correlation coefficient.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return matthews_corrcoef(targets, hard_preds)


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def roc(targets: List[float], preds: List[float]) -> float:
    try:
        return roc_auc_score(targets, preds)
    except ValueError:
        return float('nan')


def pearson_r(targets, preds):
    return pearsonr(targets.ravel(), preds.ravel())[0]


def get_metric_func(
    metric: str,
) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.
    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'roc_auc':
        return roc

    if metric == 'prc_auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'pearsonr':
        return pearson_r

    if metric == 'accuracy':
        return accuracy

    if metric == 'balanced_accuracy':
        return balanced_accuracy

    if metric == 'recall':
        return recall

    if metric == 'precision':
        return precision

    if metric == 'sensitivity':
        return sensitivity

    if metric == 'specificity':
        return specificity

    if metric == 'matthews_corrcoef':
        return mcc

    raise ValueError(f'Metric "{metric}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=30, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
