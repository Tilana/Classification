#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from lda import Preprocessor

class testPreprocessor(unittest.TestCase):

    def setUp(self):
        self.processor = Preprocessor()
        self.text = 'She is five-years old and has nine brothers and four sisters, he is eight and has four brothers.'

    def test_word2num(self):
        word2num = self.processor.word2num
        self.assertEqual(word2num('eight'), 8)
        self.assertEqual(word2num('thirteen'), 13)
        self.assertEqual(word2num('test'), None)

    def test_findNumbers(self):
        self.assertEqual(self.processor.findNumbers(self.text), set(['four', 'eight', 'nine', 'five']))


    def test_numbersInTextToDigits(self):
        target = 'She is 5-years old and has 9 brothers and 4 sisters, he is 8 and has 4 brothers.'
        self.assertEqual(self.processor.numbersInTextToDigits(self.text), target)


    def test_wordTokenize(self):
        text = 'This is a text. Split it in tokens.'
        tokens = ['This', 'is', 'a', 'text', '.', 'Split', 'it', 'in', 'tokens', '.']
        self.assertEqual(self.processor.wordTokenize(text), tokens)

    def test_splitListInChunks_n3_ov1(self):
        testList = [1,2,3,4,5,6]
        targetList = [[1,2,3], [3,4,5], [5,6]]
        self.assertListEqual(self.processor.splitListInChunks(testList, n=3, overlap=1), targetList)

    def test_splitListInChunks_n4_ov2(self):
        testList = [1,2,3,4,5,6]
        targetList = [[1,2,3,4], [3,4,5,6]]
        self.assertListEqual(self.processor.splitListInChunks(testList, n=4, overlap=2), targetList)

    def test_splitListInChunks_n3_ov2(self):
        testList = [1,2,3,4,5,6]
        targetList = [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
        self.assertListEqual(self.processor.splitListInChunks(testList, n=3, overlap=2), targetList)

    def test_splitInChunks(self):
        target = ['She is five-years old and has', 'and has nine brothers and four', 'and four sisters , he is', 'he is eight and has four', 'has four brothers .']
        self.assertListEqual(self.processor.splitInChunks(self.text, n=6, overlap=2), target)


    #def test_removeHTMLtags(self):
    #    text = '<h>title<\h> <p>text<\p>'
    #    target = 'title text'
    #    self.assertEqual(self.processor.removeHTMLtags(text), target)



if __name__ == '__main__':
    unittest.main()
