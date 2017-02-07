import unittest
import pandas as pd
from lda import FeatureExtractor

class testFeatureExtractor(unittest.TestCase):

    FeatureExtractor = FeatureExtractor()
    path = 'Documents/ICAAD/ICAAD.pkl'
    data = pd.read_pickle(path)

    def setUp(self):
        self.testDoc = self.data.loc[1450]
        

    def test_extractYear(self):
        docTitle = self.testDoc.title
        year = self.testDoc.Year
        self.assertEqual(self.FeatureExtractor.extractYear(docTitle), year)

    def test_extractCourt(self):
        docTitle = self.testDoc.title
        court = self.testDoc.Court
        self.assertEqual(self.FeatureExtractor.extractCourt(docTitle), court)


    def test_extractAge(self):
        testString = 'He was 17-years-old. He is twelve years old. At the age of 16 he started walking'
        target = ['17-years-old', 'twelve years old', 'age of 16']
        self.assertEqual(self.FeatureExtractor.extractAge(testString), target)


    def test_extractDigit(self):
        testString = '[2000]'
        self.assertEqual(self.FeatureExtractor.extractDigit(testString), 2000)

    
    


if __name__ == '__main__':
    unittest.main()
