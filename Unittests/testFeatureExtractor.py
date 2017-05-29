import unittest
import pandas as pd
from lda import FeatureExtractor


class testFeatureExtractor(unittest.TestCase):

    FeatureExtractor = FeatureExtractor()
    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)

    def setUp(self):
        self.testDoc1 = self.data.loc[1450]
        self.testDoc2 = self.data.loc[37]
        
    
    def test_year(self):
        docTitle = self.testDoc1.title
        year = self.testDoc1.Year
        self.assertEqual(self.FeatureExtractor.year(docTitle), year)

    def test_court(self):
        docTitle = self.testDoc1.title
        court = self.testDoc1.Court
        self.assertEqual(self.FeatureExtractor.court(docTitle), court)


    def test_age(self):
        testString = 'He was 17-years-old. He is 12 years old. At the age of 16 he started walking'
        target = ['17-years-old', '12 years old', 'age of 16']
        self.assertEqual(self.FeatureExtractor.age(testString), target)

    #def test_ageRange(self):
        #testString = 'she is above the age of 13 and under the age of 16'
        #target = ['above the age of 13', 'under the age of 16']
        #self.assertEqual(self.FeatureExtractor.ageRange(testString), target)

        #testString = 'She is between the age of 17 and 19. He is between 12 and 14 years old'
        #target = ['between the age of 17 and 19', 'between 12 and 14 years old']
        #self.assertEqual(self.FeatureExtractor.ageRange(testString), target)


    def test_extractDigit(self):
        testString = '[2000]'
        self.assertEqual(self.FeatureExtractor.extractDigit(testString), 2000)


    def test_caseType(self):
        self.assertEqual(self.FeatureExtractor.caseType(self.testDoc1.text), 'SENTENCE')
        self.assertEqual(self.FeatureExtractor.caseType(self.testDoc2.text), 'SUMMING UP')


    def test_unique(self):
        testList = ['a', 'c', 'a', 'c', 'b', 'b']
        target = ['a', 'c', 'b']
        self.assertEqual(self.FeatureExtractor.unique(testList), target)

    def test_getFirstElement(self):
        self.assertEqual(self.FeatureExtractor.getFirstElement(['b', 'c', 'c']), 'b')
        self.assertEqual(self.FeatureExtractor.getFirstElement([]), None)

    def test_findWordlistElem(self):
        text = 'In the family the mother and father live together with their child'
        target = ['family', 'mother', 'father', 'child']
        self.assertEqual(self.FeatureExtractor.findWordlistElem(text, 'family'), target)


    def test_groupTuples(self):
        tupleList = [('group', 'together'), ('all', 'tuple', 'elements')]
        target = ['group together', 'all tuple elements']
        self.assertEqual(self.FeatureExtractor.groupTuples(tupleList), target)
    

if __name__ == '__main__':
    unittest.main()
