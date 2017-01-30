import unittest
from lda import Evaluation

class testEvaluation(unittest.TestCase):

    def setUp(self):
        target =     [0,1,1,0,1,0]
        prediction = [0,1,0,0,1,1]
        self.evaluation = Evaluation(target, prediction)

    def test_createTags(self):
        self.evaluation.createTags()
        result = ['TN','TP','FN','TN','TP','FP']
        self.assertEqual(self.evaluation.tags, result)

    def test_setTag(self):
        self.evaluation.tags = ['TN', 'TP', 'FN', 'TN', 'TP']
        
        self.evaluation.setTag('TP')
        self.assertEqual(self.evaluation.TP, [1,4])

        self.evaluation.setTag('FP')
        self.assertEqual(self.evaluation.FP, [])

    def test_accuracy(self):
        self.evaluation.TP = [1]*25
        self.evaluation.TN = [1]*50
        self.evaluation.n = 100

        self.evaluation.accuracy()
        self.assertEqual(self.evaluation.accuracy, 0.75)

    def test_recall(self):
        self.evaluation.TP = [1]*25
        self.evaluation.FN = [1]*75
        
        self.evaluation.recall()
        self.assertEqual(self.evaluation.recall, 0.25)


    def test_precision(self):
        self.evaluation.TP = [1]*25
        self.evaluation.FP = [1]*75                       
        self.evaluation.precision()
        self.assertEqual(self.evaluation.precision, 0.25)



if __name__ == '__main__':
    unittest.main()
