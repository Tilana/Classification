import unittest
import pandas as pd
from lda import dataframeUtils as df
from pandas.util.testing import assert_frame_equal

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
        self.df2 = pd.DataFrame({'Array': [[0,2,3], [1,1,2], [7,1,3], [4,2,1,]], 'Years': [2010,2011,2010,2013], 'Strings':['a','b','c','d'], 'Array2':[[True,False], [False,False],[False, True], [False, True]]})

    def test_getRow(self):
        self.assertEqual(df.getRow(self.df, 'row name', '2nd row', ['A', 'B']), [2,5])
        self.assertEqual(df.getRow(self.df, 'row name', '2nd row', ['B']), [5])


    def test_filterData(self):
        target = self.df[0:2] 
        self.assertTrue(target.equals(df.filterData(self.df, 'C')))

    def test_getIndex(self):
        target = [0, 1, 2]
        self.assertEqual(df.getIndex(self.df), target)

    def test_splitArrayColumn(self):
        target1 = pd.DataFrame({'Years': [2010,2011,2010,2013], 'Strings':['a','b','c','d'], 0: [0,1,7,4], 1:[2,1,1,2], 2:[3,2,3,1]})
        target2 = pd.DataFrame({0:[True, False, False, False], 1:[False, False, True, True]} )
        target = pd.concat([target1, target2], axis=1)
        assert_frame_equal(df.flattenDataframe(self.df2).sort_index(axis=1), target.sort_index(axis=1)) 

    def test_arrayColumnToDataframe(self):
        target = pd.DataFrame([[0,2,3], [1,1,2], [7,1,3], [4,2,1]])
        assert_frame_equal(target, df.arrayColumnToDataframe(self.df2['Array']))


    def test_getArrayColumns(self):
        self.assertEqual(df.getArrayColumns(self.df2), ['Array', 'Array2'])



#    def test_createNumericFeature(self):
#        target = self.df
#        target.loc[:,'D'] = [0, 1, 1]
#        print target
#        print df.createNumericFeature(self.df, 'D')
#        self.assertTrue(target.equals(df.createNumericFeature(self.df, 'D')))


if __name__ == '__main__':
    unittest.main()

