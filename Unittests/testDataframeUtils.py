import unittest
import pandas as pd
import numpy as np
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
        self.df2 = pd.DataFrame({'Array': [[0,2,3], [1,1,2], [7,1,3], [4,2,1,]], 'Years': [2010,2011,2010,2013], 'Strings':['a','b','c','d'], 'BoolArray':[[True,False], [False,False],[False, True], [False, True]]})

    def test_getRow(self):
        self.assertEqual(df.getRow(self.df, 'row name', '2nd row', ['A', 'B']), [2,5])
        self.assertEqual(df.getRow(self.df, 'row name', '2nd row', ['B']), [5])


    def test_filterData(self):
        target = self.df[0:2] 
        self.assertTrue(target.equals(df.filterData(self.df, 'C')))

    def test_getIndex(self):
        target = [0, 1, 2]
        self.assertEqual(df.getIndex(self.df), target)

    def test_flattenDataframe(self):
        target1 = pd.DataFrame({'Years': [2010,2011,2010,2013], 'Strings':['a','b','c','d'], 'Array0': [0,1,7,4], 'Array1':[2,1,1,2], 'Array2':[3,2,3,1]})
        target2 = pd.DataFrame({'BoolArray0':[True, False, False, False], 'BoolArray1':[False, False, True, True]} )
        target = pd.concat([target1, target2], axis=1)
        assert_frame_equal(df.flattenDataframe(self.df2).sort_index(axis=1), target.sort_index(axis=1)) 

    def test_arrayColumnToDataframe(self):
        target = pd.DataFrame([[0,2,3], [1,1,2],[7,1,3], [4,2,1]], columns=['Array0', 'Array1', 'Array2'] )
        assert_frame_equal(target, df.arrayColumnToDataframe(self.df2, 'Array'))


    def test_getArrayColumns(self):
        self.assertEqual(df.getArrayColumns(self.df2), ['Array', 'BoolArray'])


    def test_createColumnNames(self):
        column = 'Array'
        target = ['Array0', 'Array1', 'Array2', 'Array3']
        self.assertEqual(df.createColumnNames('Array', 4), target)


    def test_combineColumnValues(self):
        target = [[0,2,3,'a'], [1,1,2,'b'], [7,1,3,'c'], [4,2,1,'d']]
        result = df.combineColumnValues(self.df2, ['Array', 'Strings'])
        self.assertEqual(result, target)

    def test_flattenArray(self):
        array = [False, 1, np.array([2,3,4]),'b', [5,6,'a']]
        target = [False,1,2,3,4,'b',5,6,'a']
        self.assertEqual(df.flattenArray(array), target)


#    def test_createNumericFeature(self):
#        target = self.df
#        target.loc[:,'D'] = [0, 1, 1]
#        print target
#        print df.createNumericFeature(self.df, 'D')
#        self.assertTrue(target.equals(df.createNumericFeature(self.df, 'D')))


if __name__ == '__main__':
    unittest.main()

