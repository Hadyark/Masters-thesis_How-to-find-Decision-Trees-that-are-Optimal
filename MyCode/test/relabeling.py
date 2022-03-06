import sys
import unittest

import numpy as np
import pandas
import pandas as pd

from MyCode import relabeling
from MyCode.relabeling import Leaf
import sys


class TestRelabeling(unittest.TestCase):

    def test_leaf(self):
        leaf = Leaf(None, 1 / 20, 1 / 20, 1 / 20, 0)
        leaf.accuracy(1 / 2, 1 / 2)

        self.assertEqual(leaf.acc, -1 / 20)
        self.assertEqual(leaf.disc, -1 / 10)

    def test_discrimination(self):
        df = pd.DataFrame({'Class': [], 'Pred': [], 'Sensitive': []})

        for i in range(0, 5):
            df.loc[len(df.index)] = [0, 0, 1]
        for i in range(0, 1):
            df.loc[len(df.index)] = [0, 1, 1]
        for i in range(0, 1):
            df.loc[len(df.index)] = [1, 0, 1]
        for i in range(0, 3):
            df.loc[len(df.index)] = [1, 1, 1]

        for i in range(0, 3):
            df.loc[len(df.index)] = [0, 0, 0]
        for i in range(0, 1):
            df.loc[len(df.index)] = [0, 1, 0]
        for i in range(0, 1):
            df.loc[len(df.index)] = [1, 0, 0]
        for i in range(0, 5):
            df.loc[len(df.index)] = [1, 1, 0]

        self.assertEqual(len((df[(df['Class'] == 0) & (df['Pred'] == 0)])), 8)
        self.assertEqual(len((df[(df['Class'] == 0) & (df['Pred'] == 1)])), 2)
        self.assertEqual(len((df[(df['Class'] == 1) & (df['Pred'] == 0)])), 2)
        self.assertEqual(len((df[(df['Class'] == 1) & (df['Pred'] == 1)])), 8)

        self.assertEqual(len((df[(df['Sensitive'] == 0)])), 10)
        self.assertEqual(len((df[(df['Sensitive'] == 0)])), 10)
        self.assertEqual(len(df), 20)

        disc = relabeling.discrimination(df['Class'], df['Pred'], df['Sensitive'])
        d1 = ((1/20) + (5/20)) / (1/2)
        d2 = ((1/20) + (3/20)) / (1/2)
        self.assertEqual(disc, d1 - d2)

if __name__ == '__main__':
    unittest.main()
