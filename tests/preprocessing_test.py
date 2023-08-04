from src.preprocessing import *
from constants import *
import unittest
import random


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


class TestOneHotEncodeDecode(unittest.TestCase):
    series1 = pd.Series([random.choice(CLASSES) for _ in range(1000)])
    series2 = pd.Series([random.choice(CLASSES) for _ in range(200)])

    def test1(self):
        encoded = one_hot_encode(self.series1)
        decoded = one_hot_decode(encoded, self.series2)
        self.assertTrue(np.equal(self.series1, decoded).all())

    def test2(self):
        encoded = one_hot_encode(self.series2)
        decoded = one_hot_decode(encoded, self.series1)
        self.assertTrue(np.equal(self.series2, decoded).all())


class TestEnumerateEncodeDecode(unittest.TestCase):
    series1 = pd.Series([random.choice(CLASSES) for _ in range(1000)])
    series2 = pd.Series([random.choice(CLASSES) for _ in range(200)])

    def test1(self):
        encoded = enumerate_encode(self.series1)
        decoded = enumerate_decode(encoded, self.series2)
        self.assertTrue(np.equal(self.series1, decoded).all())

    def test2(self):
        encoded = enumerate_encode(self.series2)
        decoded = enumerate_decode(encoded, self.series1)
        self.assertTrue(np.equal(self.series2, decoded).all())


if __name__ == '__main__':
    unittest.main()
