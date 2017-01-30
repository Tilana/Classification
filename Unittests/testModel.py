import unittest
from lda import Model, Document, Topic, Info

class testModel(unittest.TestCase):

    def test_tupleToTopicList(self):
        info = Info()
        info.modelType= 'LDA'
        info.numberTopics = 3
        info.categories = []
        model = Model(info)
        wordDistribution1 = [(u'Test', 0.45), (u'Topic', 0.003), (u'List', -0.01)]
        wordDistribution2 = [(u'List', 0.0021), (u'2', -0.001)]
        topicList = [(0, wordDistribution1), (1, wordDistribution2)]

        topic1 = Topic()
        topic1.number = 0
        topic1.wordDistribution = wordDistribution1

        topic2 = Topic()
        topic2.number = 1
        topic2.wordDistribution = wordDistribution2
        
        testList = model._tupleToTopicList(topicList)
        targetList = [topic1, topic2]
        for ind,topics in enumerate(targetList):
            self.assertEqual(targetList[ind].number, testList[ind].number)
            self.assertEqual(targetList[ind].wordDistribution, testList[ind].wordDistribution)


if __name__ == '__main__':
    unittest.main()
