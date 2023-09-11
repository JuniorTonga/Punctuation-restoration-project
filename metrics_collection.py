import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def masked_accuracy(y_true, y_hat, y_mask):
    y_true = y_true[y_mask == 1].flatten()
    y_hat = y_hat.argmax(axis=-1)[y_mask == 1].flatten()

    return np.sum(y_hat == y_true) / y_true.shape[0]


def masked_f1(y_true, y_hat, y_mask, score="f1", average="macro", labels=[1, 2, 3]):
    y_true = y_true[y_mask == 1]
    y_hat = y_hat.argmax(axis=-1)[y_mask == 1]
    if score == "f1":
        return f1_score(y_true, y_hat, average=average, labels=labels, zero_division=0)
    elif score == "recall":
        return recall_score(
            y_true, y_hat, average=average, labels=labels, zero_division=0
        )
    elif score == "precision":
        return precision_score(
            y_true, y_hat, average=average, labels=labels, zero_division=0
        )


accuracy_metric = {"name": "accuracy", "fn": masked_accuracy}

f1_metric = {"name": "f1-score", "fn": masked_f1}
