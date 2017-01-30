import unittest
import pandas as pd
from lda import dataframeUtils as df

class testDataframeUtils(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'row name': ['1st row', '2nd row', '3rd row'],
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [True, True, False],
            'D': ['<5', '>5', '<5'],
            'E': ['red', 'green', 'green']},
            columns=['row name', 'A', 'B', 'C', 'D', 'E']) 


    def test_getRow(self):
        self.assertEqual(df.getRow(self.df, 'row name', '2nd row', ['A', 'B']), [2,5])
        self.assertEqual(df.getRow(self.df, 'row name', '2nd row', ['B']), [5])


    def test_filterData(self):
        target = self.df[0:2] 
        self.assertTrue(target.equals(df.filterData(self.df, 'C')))

    def test_getIndex(self):
        target = [0, 1, 2]
        self.assertEqual(df.getIndex(self.df), target)

#    def test_createNumericFeature(self):
#        target = self.df
#        target.loc[:,'D'] = [0, 1, 1]
#        print target
#        print df.createNumericFeature(self.df, 'D')
#        self.assertTrue(target.equals(df.createNumericFeature(self.df, 'D')))


if __name__ == '__main__':
    unittest.main()

