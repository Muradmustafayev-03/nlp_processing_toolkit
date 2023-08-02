import pandas as pd
import numpy as np
import spacy

# run 'python -m spacy download en_core_web_sm' in the terminal before running this line
nlp = spacy.load("en_core_web_sm")


def filter_tokens(text: str) -> str:
    """
    Filter out punctuation and stop tokens.
    :param text: input text (sentence)
    :return: text without punctuation and stop words
    """
    doc = nlp(text)
    filtered = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(filtered)


def extract_lemma(text: str) -> str:
    """
    Extract lemma(base) from each word in the text.
    Example: feeling -> feel; pencils -> pencil; exhausted -> exhaust
    :param text: input text (sentence)
    :return: text with each word replaced to its base word
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def filter_and_extract_lemma(text: str) -> str:
    """
    Filter out punctuation and stop tokens and extract lemma(base) from each word in the text.
    :param text: input text (sentence)
    :return: text without punctuation and stop words and each word replaced to its base word
    """
    doc = nlp(text)
    filtered = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(filtered)


def one_hot_encode(column: pd.Series) -> np.array:
    """
    Converts a data series of classes into one-hot encoded array
    :param column: the data Series or DataFrame column to encode
    :return: 2D array with encoded data
    """
    return np.array(pd.get_dummies(column))


def one_hot_decode(encoded: np.array or list, classes: pd.Series or list or set):
    if isinstance(encoded, list):
        encoded = np.array(encoded)
    if isinstance(classes, pd.Series):
        classes = classes.values.tolist()
    if isinstance(classes, list):
        classes = set(classes)
    if isinstance(classes, set):
        classes = sorted(list(classes))

    return classes[np.argmax(encoded)]
