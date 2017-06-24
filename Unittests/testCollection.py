import unittest
from lda import Collection
import numpy as np
import pandas as pd

class testCollection(unittest.TestCase):

    def setUp(self):
        self.collection = Collection('Unittests/testData.csv')

    def test_init_empty(self):
        collection = Collection()
        self.assertTrue(collection.data.empty)
        self.assertEqual(collection.nrDocs, 0)

    def test_init_data(self):
        self.assertFalse(self.collection.data.empty)
        self.assertGreater(self.collection.nrDocs, 0)

    def test_cleanDataframe(self):
        self.collection.cleanDataframe()
        self.assertTrue(self.collection.nrDocs, 3)
        nanColumns = self.collection.data.isnull().all()
        for col in nanColumns:
            self.assertFalse(col)
        nanRows = self.collection.data.isnull().all(axis=1)
        for col in nanRows:
            self.assertFalse(col)

        

if __name__ == '__main__':
    unittest.main()
