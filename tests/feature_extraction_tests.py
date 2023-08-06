from src.feature_extraction import *
from test_constants import *
import unittest


class TestFilterTokens(unittest.TestCase):
    def test_sentences(self):
        result = [filter_tokens(sentence, True) for sentence in SENTENCES]
        self.assertEqual(FILTERED, result)

    def test_filtered(self):
        result = [filter_tokens(sentence, True) for sentence in FILTERED]
        self.assertEqual(FILTERED, result)

    def test_lemmas(self):
        result = [filter_tokens(sentence, True) for sentence in LEMMAS]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_punctuation(self):
        for case in PUNCTUATION:
            self.assertEqual('', filter_tokens(case, True))


class TestExtractLemma(unittest.TestCase):
    def test_sentences(self):
        result = [extract_lemma(sentence, True) for sentence in SENTENCES]
        self.assertEqual(LEMMAS, result)

    def test_filtered(self):
        result = [extract_lemma(sentence, True) for sentence in FILTERED]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_lemmas(self):
        result = [extract_lemma(sentence, True) for sentence in LEMMAS]
        self.assertEqual(LEMMAS, result)


class TestFilterAndExtractLemma(unittest.TestCase):
    def test_sentences(self):
        result = [filter_and_extract_lemma(sentence, True) for sentence in SENTENCES]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_filtered(self):
        result = [filter_and_extract_lemma(sentence, True) for sentence in FILTERED]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_lemmas(self):
        result = [filter_and_extract_lemma(sentence, True) for sentence in LEMMAS]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_punctuation(self):
        for case in PUNCTUATION:
            self.assertEqual('', filter_tokens(case, True))
