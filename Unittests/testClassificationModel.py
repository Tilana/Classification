import unittest
from lda import ClassificationModel
import pandas as pd

class testClassificationModel(unittest.TestCase):

    def setUp(self):
        self.model = ClassificationModel()
        self.model.data = pd.DataFrame({'Binary': [True, True, False, True], 'Years': [2010,2011,2010,2013], 'Strings':['a','b','c','d']})

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
        



    def determineClassificationType(self):
        return True
        


if __name__ == '__main__':
    unittest.main()
