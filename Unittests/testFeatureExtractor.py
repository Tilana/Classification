import unittest
import pandas as pd
from lda import FeatureExtractor

class testFeatureExtractor(unittest.TestCase):

    FeatureExtractor = FeatureExtractor()
    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)

    def setUp(self):
        self.testDoc = self.data.loc[1450]
        
    
    def test_year(self):
        docTitle = self.testDoc.title
        year = self.testDoc.Year
        self.assertEqual(self.FeatureExtractor.year(docTitle), year)

    def test_court(self):
        docTitle = self.testDoc.title
        court = self.testDoc.Court
        self.assertEqual(self.FeatureExtractor.court(docTitle), court)


    def test_age(self):
        testString = 'He was 17-years-old. He is twelve years old. At the age of 16 he started walking'
        target = ['17-years-old', 'twelve years old', 'age of 16']
        self.assertEqual(self.FeatureExtractor.age(testString), target)


    def test_extractDigit(self):
        testString = '[2000]'
        self.assertEqual(self.FeatureExtractor.extractDigit(testString), 2000)


    def test_caseType(self):
        self.assertEqual(self.FeatureExtractor.caseType(self.testDoc.text), 'SENTENCE')
        print 1
    

if __name__ == '__main__':
    unittest.main()
