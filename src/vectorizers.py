from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


class ExtendedTokenizer(Tokenizer):
    """
    Extended keras sparse tokenizer with additional methods for converting texts to padded sequences and back.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def texts_to_padded_sequences(self, texts, maxlen=None):
        """
        Convert texts to padded sequences.
        :param texts: texts to vectorize
        :param maxlen: maximum length of the sequences
        :return: padded sequences
        """
        return pad_sequences(self.texts_to_sequences(texts), maxlen=maxlen)

    def padded_sequences_to_text(self, sequences):
        """
        Convert padded sequences to texts.
        :param sequences: sequences to convert
        :return: texts
        """
        cleared_sequences = [list(filter(lambda x: x != 0, row)) for row in sequences]
        return self.sequences_to_texts(cleared_sequences)


class Word2VecVectorizer(Word2Vec):
    """
    Extended Word2Vec model with additional methods to vectorize sentences.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def vectorize_sentence(self, sentence: str) -> np.ndarray:
        """
        Vectorize a sentence.
        :param sentence: sentence to vectorize
        :return: vectorized sentence
        """
        vectorized = [self.wv[word] for word in sentence.split(' ') if word in self.wv]
        if not vectorized:
            # If none of the words are in the vocabulary, return zeros
            vectorized = [np.zeros(self.vector_size)]
        return np.mean(vectorized, axis=0)

    def vectorize_sentences(self, sentences: pd.Series) -> np.ndarray:
        """
        Vectorize a series of sentences.
        :param sentences: sentences to vectorize
        :return: vectorized sentences
        """
        return np.array([self.vectorize_sentence(sentence) for sentence in sentences.values])
