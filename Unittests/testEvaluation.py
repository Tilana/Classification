import unittest
from lda import Evaluation

class testEvaluation(unittest.TestCase):

    def setUp(self):
        target =     [0,1,1,0,1,0,0,0,1,1]
        prediction = [0,1,0,0,1,0,0,1,1,1]
        self.evaluation = Evaluation(target, prediction)

    def test_createTags(self):
        self.evaluation.createTags()
        result = ['TN','TP','FN','TN','TP','TN', 'TN', 'FP', 'TP','TP']
        self.assertEqual(self.evaluation.tags, result)

    def test_setTag(self):
        self.evaluation.tags = ['TN', 'TP', 'FN', 'TN', 'TP']
        
        self.evaluation.setTag('TP')
        self.assertEqual(self.evaluation.TP, [1,4])

        self.evaluation.setTag('FP')
        self.assertEqual(self.evaluation.FP, [])

    def test_accuracy(self):
        self.evaluation.accuracy()
        self.assertEqual(self.evaluation.accuracy, 0.8)

    def test_recall(self):
        self.evaluation.recall()
        self.assertEqual(self.evaluation.recall, 0.8)


    def test_precision(self):
        self.evaluation.precision()
        self.assertEqual(self.evaluation.precision, 0.8)




if __name__ == '__main__':
    unittest.main()
