#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from lda import Preprocessor 

class testPreprocessor(unittest.TestCase):

    def setUp(self):
        self.processor = Preprocessor()

    def test_word2num(self):
        word2num = self.processor.word2num
        self.assertEqual(word2num('eight'), 8)
        self.assertEqual(word2num('thirteen'), 13)
        self.assertEqual(word2num('test'), None)

    def test_findNumbers(self):
        text = 'She is five-years old and has sixteen brothers and four sisters, he is eight and has four brothers.'
        print self.processor.findNumbers(text)
        self.assertEqual(self.processor.findNumbers(text), set(['four', 'eight', 'sixteen', 'five']))


    def test_numbersInTextToDigits(self):
        text = 'She is four-years old and has nine brothers, he is eight and has four brothers.'
        target = 'She is 4-years old and has 9 brothers, he is 8 and has 4 brothers.'
        self.assertEqual(self.processor.numbersInTextToDigits(text), target)


    def test_wordTokenize(self):
        text = 'This is a text. Split it in tokens.'
        tokens = ['This', 'is', 'a', 'text', '.', 'Split', 'it', 'in', 'tokens', '.'] 
        self.assertEqual(self.processor.wordTokenize(text), tokens)


if __name__ == '__main__':
    unittest.main()
