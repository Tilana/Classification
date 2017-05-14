import unittest
from lda import Evaluation
import pandas as pd 
from pandas.util.testing import assert_frame_equal

class testEvaluation(unittest.TestCase):

    def setUp(self):
        binaryTarget =     [0,1,1,0,1,0,0,0,1,1]
        binaryPrediction = [0,1,0,0,1,0,0,1,1,1]
        self.bin_evaluation = Evaluation(binaryTarget, binaryPrediction, 'binary')

        multiTarget = [1,3,3,2,1,1,2,1]
        multiPrediction = [1,2,3,2,2,3,2,3]
        self.multi_evaluation = Evaluation(multiTarget, multiPrediction, 'macro')

    def test_createTags(self):
        self.bin_evaluation.createTags()
        bin_tags= ['TN','TP','FN','TN','TP','TN', 'TN', 'FP', 'TP','TP']
        self.assertEqual(self.bin_evaluation.tags, bin_tags)

        self.multi_evaluation.createTags()
        multi_tags = ['T','F','T','T','F','F','T','F']
        self.assertEqual(self.multi_evaluation.tags, multi_tags)

    def test_setTag(self):
        self.bin_evaluation.tags = ['TN', 'TP', 'FN', 'TN', 'TP']
        
        self.bin_evaluation.setTag('TP')
        self.assertEqual(self.bin_evaluation.TP, [1,4])

        self.bin_evaluation.setTag('FP')
        self.assertEqual(self.bin_evaluation.FP, [])

    def test_accuracy(self):
        self.bin_evaluation.accuracy()
        self.assertEqual(self.bin_evaluation.accuracy, 0.8)

        self.multi_evaluation.accuracy()
        self.assertEqual(self.multi_evaluation.accuracy, 0.5)

    def test_recall(self):
        self.bin_evaluation.recall()
        self.assertEqual(self.bin_evaluation.recall, 0.8)


    def test_precision(self):
        self.bin_evaluation.precision()
        self.assertEqual(self.bin_evaluation.precision, 0.8)


    def test_confusionMatrix(self):
        self.bin_evaluation.confusionMatrix()
        bin_confusionMatrix = pd.DataFrame([[4,1],[1,4]])
        assert_frame_equal(self.bin_evaluation.confusionMatrix, bin_confusionMatrix)

        self.multi_evaluation.confusionMatrix()
        multi_confusionMatrix = pd.DataFrame([[1,1,2],[0,2,0],[0,1,1]])
        assert_frame_equal(self.multi_evaluation.confusionMatrix, multi_confusionMatrix)

    def test_normalizeMatrix(self):
        self.bin_evaluation.confusionMatrix = pd.DataFrame([[2,8],[9,3]])
        normalizedMatrix = pd.DataFrame([[0.2, 0.8],[0.75, 0.25]])
        self.bin_evaluation.normalizeMatrix()
        assert_frame_equal(self.bin_evaluation.normConfusionMatrix, normalizedMatrix)
        




if __name__ == '__main__':
    unittest.main()
