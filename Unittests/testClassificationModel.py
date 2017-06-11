import unittest
from lda import ClassificationModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
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

    def test_buildClassifier(self):
        self.model.buildClassifier('DecisionTree')
        self.assertDictEqual(self.model.classifier.__dict__, DecisionTreeClassifier().__dict__)
        params = {'min_samples_leaf': [2,5], 'max_depth':[3,5,7], 'criterion':['gini','entropy']}
        self.assertEqual(self.model.parameters, params) 

        self.model.buildClassifier('MultinomialNB')
        self.assertDictEqual(self.model.classifier.__dict__, MultinomialNB().__dict__)
        params = {'alpha':[0,0.01, 0.3, 0.6, 1], 'fit_prior':[True, False]}
        self.assertEqual(self.model.parameters, params)

    def test_buildParamClassifier(self):
        classifierType = 'MultinomialNB'
        params = {'alpha':0.01, 'fit_prior':False}
        self.model.classifier = MultinomialNB()
        self.model.buildParamClassifier(params)
        self.assertDictContainsSubset(params, self.model.classifier.get_params())

    def test_getWordIndex(self):
        self.model.vocabulary = ['word', 'word pair', 'new word']
        self.assertEqual(self.model.getWordIndex('new word'), 2)
        self.assertEqual(self.model.getWordIndex('not in vocab'), None)
        

    def test_increaseWeights(self):
        self.model.vocabulary = ['word', 'word pair', 'new word']
        vocabLength = len(self.model.vocabulary)
        nrRows = self.model.data.shape[0]
        originalWeights = np.random.rand(nrRows, vocabLength)
        self.model.data['tfIdf'] = list(originalWeights)
        whiteList = ['word pair', 'word', 'not in vocab']
        self.model.increaseWeights(self.model.data, 'tfIdf', whiteList)
        
        for index, newWeights in self.model.data['tfIdf'].iteritems():
            self.assertGreater(newWeights[0], originalWeights[index,0])
            self.assertGreater(newWeights[1], originalWeights[index,1]) 
            self.assertEqual(newWeights[2], originalWeights[index,2]) 


    def test_proportionalValue(self):
        w1 = np.random.uniform(1,2,[1,5])
        w2 = np.random.uniform(0,1,[1,5])
        weights = np.vstack([w1,w2])
        addValue = self.model.proportionalValue(weights)
        self.assertGreater(addValue[0], addValue[1])
        self.assertEqual(addValue.shape, (2,))

if __name__ == '__main__':
    unittest.main()
