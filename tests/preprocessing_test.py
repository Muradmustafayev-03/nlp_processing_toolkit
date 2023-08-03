from src.preprocessing import *
import unittest
import random

SENTENCES = [
    'The sun set over the horizon, casting a golden hue on the tranquil sea.',
    'Birds chirped merrily in the treetops, heralding the arrival of a new day.',
    'With a gentle breeze rustling the leaves, the scent of blooming flowers filled the air.',
    'Laughter echoed through the park as children played games and enjoyed the warm summer afternoon.',
    'In the distance, a faint rumble of thunder warned of an approaching storm.'
]

FILTERED = [
    'sun set horizon casting golden hue tranquil sea',
    'birds chirped merrily treetops heralding arrival new day',
    'gentle breeze rustling leaves scent blooming flowers filled air',
    'laughter echoed park children played games enjoyed warm summer afternoon',
    'distance faint rumble thunder warned approaching storm'
]

LEMMAS = [
    'the sun set over the horizon , cast a golden hue on the tranquil sea .',
    'bird chirp merrily in the treetop , herald the arrival of a new day .',
    'with a gentle breeze rustle the leaf , the scent of bloom flower fill the air .',
    'laughter echo through the park as child play game and enjoy the warm summer afternoon .',
    'in the distance , a faint rumble of thunder warn of an approach storm .'
]

LEMMAS_FILTERED = [
    'sun set horizon cast golden hue tranquil sea',
    'bird chirp merrily treetop herald arrival new day',
    'gentle breeze rustle leaf scent bloom flower fill air',
    'laughter echo park child play game enjoy warm summer afternoon',
    'distance faint rumble thunder warn approach storm'
]

PUNCTUATION = [
    '!!!???!!!????!!!!!!!??????!!!!!!',
    '!&*()',
    ',. ;: " ? []',
    '[].{}()'
    ''
]

CLASSES = ['ClassA', 'ClassB', 'ClassC', 'ClassD', 'ClassE']


class TestFilterTokens(unittest.TestCase):
    def test_sentences(self):
        result = [filter_tokens(sentence) for sentence in SENTENCES]
        self.assertEqual(FILTERED, result)

    def test_filtered(self):
        result = [filter_tokens(sentence) for sentence in FILTERED]
        self.assertEqual(FILTERED, result)

    def test_lemmas(self):
        result = [filter_tokens(sentence) for sentence in LEMMAS]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_punctuation(self):
        for case in PUNCTUATION:
            self.assertEqual('', filter_tokens(case))


class TestExtractLemma(unittest.TestCase):
    def test_sentences(self):
        result = [extract_lemma(sentence) for sentence in SENTENCES]
        self.assertEqual(LEMMAS, result)

    def test_filtered(self):
        result = [extract_lemma(sentence) for sentence in FILTERED]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_lemmas(self):
        result = [extract_lemma(sentence) for sentence in LEMMAS]
        self.assertEqual(LEMMAS, result)


class TestFilterAndExtractLemma(unittest.TestCase):
    def test_sentences(self):
        result = [filter_and_extract_lemma(sentence) for sentence in SENTENCES]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_filtered(self):
        result = [filter_and_extract_lemma(sentence) for sentence in FILTERED]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_lemmas(self):
        result = [filter_and_extract_lemma(sentence) for sentence in LEMMAS]
        self.assertEqual(LEMMAS_FILTERED, result)

    def test_punctuation(self):
        for case in PUNCTUATION:
            self.assertEqual('', filter_tokens(case))


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


if __name__ == '__main__':
    unittest.main()
