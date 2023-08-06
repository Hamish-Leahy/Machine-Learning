import numpy as np

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of a binary classification model.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

def precision(y_true, y_pred):
    """
    Calculate the precision of a binary classification model.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: Precision score.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true, y_pred):
    """
    Calculate the recall of a binary classification model.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: Recall score.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred):
    """
    Calculate the F1-score of a binary classification model.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: F1-score.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def roc_auc(y_true, y_prob):
    """
    Calculate the ROC-AUC score of a binary classification model.

    Args:
        y_true (numpy.ndarray): True labels.
        y_prob (numpy.ndarray): Predicted probabilities.

    Returns:
        float: ROC-AUC score.
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_prob)
