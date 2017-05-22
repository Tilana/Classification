import unittest
from lda import ClassificationModel
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

class testClassificationModel(unittest.TestCase):

    def setUp(self):
        self.model = ClassificationModel()
        self.model.data = pd.DataFrame({'Binary': [True, True, False, True], 'Years': [2010,2011,2010,2013], 'Strings':['a','b','c','d'], 'Arrays':[[1,2],[5,1],[3,2],[4,1]]})

    def test_getFeatureList(self):
        properties = ['Binary']
        features = [[True], [True], [False], [True]]
        assert_array_equal(self.model.getFeatureList(self.model.data, properties), features)

        properties = ['Binary', 'Arrays']
        features = [[True, 1, 2], [True,5,1], [False,3,2], [True,4,1]]
        self.model.getFeatureList(self.model.data, properties)
        assert_array_equal(self.model.getFeatureList(self.model.data, properties), features)

    def test_setTargetLabels(self):
        self.model.setTargetLabels('Binary') 
        self.assertEqual(self.model.targetLabels, [True, False])

        labels = self.model.setTargetLabels('Years') 
        self.assertEqual(self.model.targetLabels, [2010, 2011, 2013])

        labels = self.model.setTargetLabels('Strings') 
        self.assertEqual(self.model.targetLabels, ['a', 'b', 'c', 'd'])

    def test_setClassificationType(self):
        self.model.setClassificationType('Binary')
        self.assertEqual(self.model.classificationType, 'binary')

        self.model.setClassificationType('Years')
        self.assertEqual(self.model.classificationType, 'multi')

    def test_createTarget_binary(self):
        self.model.targetFeature = 'Binary'
        self.model.createTarget()
        self.assertEqual(self.model.classificationType, 'binary')
        self.assertEqual(self.model.target.tolist(), [True, True, False, True])
        self.assertEqual(self.model.targetLabels, [True, False])

    def test_createTarget_multi(self):
        self.model.targetFeature = 'Strings'
        self.model.createTarget()
        self.assertEqual(self.model.classificationType, 'multi')
        self.assertEqual(self.model.target.tolist(), [0,1,2,3])
        self.assertEqual(self.model.targetLabels, ['a', 'b', 'c', 'd'])


    def test_setEvaluationAverage_binary(self):
        self.model.classificationType = 'binary'
        self.model.setEvaluationAverage()
        self.model.evaluationAverage = 'binary'


    def test_setEvaluationAverage_multiDefault(self):
        self.model.classificationType = 'multi'       
        self.model.setEvaluationAverage()
        self.model.evaluationAverage = 'macro'

    
    def test_setEvaluationAverage_multiSelfSet(self):
        self.model.classificationType = 'binary'
        self.model.setEvaluationAverage('weighted')
        self.model.evaluationAverage = 'weighted'

        
if __name__ == '__main__':
    unittest.main()
