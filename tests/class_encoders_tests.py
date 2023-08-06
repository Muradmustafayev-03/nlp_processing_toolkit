from src.class_encoders import *
from test_constants import *
import unittest
import random


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
