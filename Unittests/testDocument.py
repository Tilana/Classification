#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from lda import Document
from lda import Entities

class testDocument(unittest.TestCase):
    
    def setUp(self):
        self.targetDocument = Document('','')
        self.testDocument = Document('Test Doc', 'Test of tokenization\n dates like 12.03.1998, 103/78 and World Health Organisation should be kept together. Words appear more more often!?')
        self.stoplist = ['and', 'of']
        self.specialChars = r'.*[\.,/?!].*'

    def test_lemmatizeTokens(self):
        testDocument = Document('','')
        testDocument.tokens = set(['children', 'forced', 'trafficking', 'prisons', 'arrested',  'United Nations', '12.03.1992', 'are', 'violations', 'bags of words'])
        testDocument.lemmatizeTokens()

        self.targetDocument.tokens = ['child', 'United Nations', '12.03.1992', 'be', 'violation', 'force', 'arrest', 'traffic', 'prison', 'bags of words']

        self.assertEqual(set(testDocument.tokens), set(self.targetDocument.tokens))

    def test_findSpecialCharacterTokens(self):
        testDocument = Document('', '')
        testDocument.tokens = ['child`s', '23.09.1998', 'test entity', 'normal', '$200 000', '809/87', 'http://asfd.org', 'talib@n?', 'end of line.\n', '.']
        specialChars = r'.*[@./,:$©].*'
        testDocument.findSpecialCharacterTokens(specialChars)

        targetDocument = Document('','')
        targetDocument.specialCharacters = ['23.09.1998', '$200 000', '809/87', 'http://asfd.org', 'talib@n?', 'end of line.\n', '.']
        self.assertEqual(set(targetDocument.specialCharacters), set(testDocument.specialCharacters))
    
    def test_removeSpecialCharacters(self):
        testDocument = Document('', '')
        testDocument.tokens = ['child`s', '23.09.1998', 'test entity', 'normal', '$200 000', '809/87', 'http://asfd.org', '809/87', 'talib@n?', '.', 'end of line.\n']
        testDocument.specialCharacters = ['23.09.1998', '$200 000', '809/87', 'http://asfd.org', 'talib@n?', '.']

        target= set(['child`s', '809/87', 'test entity', 'normal', 'end of line.\n'])
        testDocument.removeSpecialCharacters()
        self.assertEqual(target, set(testDocument.tokens))

    def test_createTokens(self):
        testDocument = Document('Test Doc', 'Test of tokenization\n dates like 12.03.1998, 103/78 and Words should be lowered and appear more more often.?')
        testDocument.createTokens()
        self.targetDocument.tokens = ['test','of','tokenization','dates','like','12.03.1998',',','103/78', 'and', 'words', 'should', 'be', 'lowered', 'and', 'appear', 'more', 'more', 'often', '.', '?']
        self.assertEqual(testDocument.tokens, self.targetDocument.tokens)


    def test_prepareDocument(self):
        targetDocument = Document('','')
        testDocument = Document('Test Doc', 'Test of tokenization\n remove to short words and spe?cial char/s, words not in whitelist or in stoplist') 
        stoplist = ['and', 'of', 'stoplist']
        specialChars = r'.*[?/].*'
        whiteList = ['test', 'of', 'tokenization', 'remove', 'to', 'short', 'word', 'spec?cial', 'char/s', 'in', 'stoplist']
        testDocument.prepareDocument(lemmatize=True, includeEntities=False, stopwords=stoplist, specialChars=specialChars, removeShortTokens=True, threshold=2, whiteList=whiteList)

        targetDocument.tokens = ['test', 'tokenization','remove','short', 'word', 'word', ]
        self.assertEqual(testDocument.tokens, targetDocument.tokens)


    def test_appendEntities(self):
        testDocument = Document('Test Document','Name entities like World Health Organization, person names like Sir James and Ms Rosa Wallis but also world locations or states like Lebanon, United States of America, Lebanon or new cities like New York have to be recognized')
        testDocument.createEntities()
        testDocument.createTokens()
        testDocument.appendEntities()

        self.targetDocument.tokens = ['name', 'entities', 'like', ',', 'person', 'names', 'like', 'sir', 'and', 'ms', 'but', 'also', 'world', 'locations', 'or', 'like', ',', 'states', ',', 'or', 'cities', 'like', 'new', 'have', 'to', 'be', 'recognized', 'world health organization', 'lebanon', 'lebanon', 'united states of america', 'new york', 'james', 'rosa wallis']
        self.assertEqual(self.targetDocument.tokens, testDocument.tokens)

    
    def test_createEntities(self):
        self.targetDocument.createEntities()
        testDocument = Document('Test Document','Name entities like World Health Organization, person names like Sir James and Ms Rosa Wallis but also locations like Lebanon, United States of America, Lebanon or cities like New York have to be recognized')
        testDocument.createEntities()
       
        self.targetDocument.entities.LOCATION = [(u'lebanon', 2), (u'united states of america', 1), (u'new york', 1)]
        self.targetDocument.entities.PERSON = [(u'james', 1), (u'rosa wallis', 1)]
        self.targetDocument.entities.ORGANIZATION = [(u'world health organization',1)]

        self.assertEqual(testDocument.entities.PERSON, self.targetDocument.entities.PERSON)
        self.assertEqual(testDocument.entities.LOCATION, self.targetDocument.entities.LOCATION)
        self.assertEqual(testDocument.entities.ORGANIZATION, self.targetDocument.entities.ORGANIZATION)


    def test_correctTokenOccurance(self):
        testDocument = Document('Test Document', 'In the world many organizations like the World Health Organization or the Union of the World exist')
        testDocument.tokens = ['world', 'many', 'organizations', 'like', 'world', 'health', 'organization', 'union', 'world', 'exist', 'world health organization', 'union of the world']
        entity = ('union of the world', 1)

        targetTokens = ['many', 'organizations', 'like', 'world', 'health', 'organization', 'world', 'exist', 'world health organization', 'union of the world']

        testDocument.correctTokenOccurance(entity[0])
        self.assertEqual(targetTokens, testDocument.tokens)


if __name__ == '__main__':
    unittest.main()
