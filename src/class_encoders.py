import pandas as pd
import numpy as np


def one_hot_encode(column: pd.Series) -> np.array:
    """
    Converts a data series of classes into one-hot encoded array
    :param column: the data Series or DataFrame column to encode
    :return: 2D array with encoded data
    """
    return np.array(pd.get_dummies(column))


def one_hot_decode(encoded: np.array, classes: pd.Series) -> np.array:
    """
    Decodes one-hot encoded values according to existing classes
    :param encoded: one-hot encoded numpy array, either from one_hot_encode function or from the model output
    :param classes: classes column of the source dataset
    :return: numpy array with the name of the class corresponding to each one-hot subarray
    """
    classes = sorted(classes.unique().tolist())
    return np.array([classes[np.argmax(one_hot)] for one_hot in encoded])


def enumerate_encode(column: pd.Series) -> np.array:
    """
    Converts a data series of classes into enumerated array
    :param column: the data Series or DataFrame column to encode
    :return: 2D array with encoded data
    """
    classes = sorted(column.unique().tolist())
    return np.array([classes.index(value) for value in column])


def enumerate_decode(encoded: np.array, classes: pd.Series) -> np.array:
    """
    Decodes enumerated values according to existing classes
    :param encoded: enumerated numpy array, either from enumerate_encode function or from the model output
    :param classes: classes column of the source dataset
    :return: numpy array with the name of the class corresponding to each enumerated value
    """
    classes = sorted(classes.unique().tolist())
    return np.array([classes[value] for value in encoded])


def one_hot_to_enumerate(encoded: np.array) -> np.array:
    """
    Converts one-hot encoded array into enumerated array
    :param encoded: one-hot encoded numpy array, either from one_hot_encode function or from the model output
    :return: enumerated numpy array
    """
    return np.array([np.argmax(one_hot) for one_hot in encoded])


def enumerate_to_one_hot(encoded: np.array, classes: pd.Series) -> np.array:
    """
    Converts enumerated array into one-hot encoded array
    :param encoded: encoded 1-D numpy array, either from enumerate_encode function or from the model output
    :param classes: classes column of the source dataset
    :return: one-hot encoded numpy array
    """
    N = len(classes.unique().tolist())
    ret = np.zeros((len(encoded), N))
    for i, value in enumerate(encoded):
        ret[i, value] = 1
    return ret
