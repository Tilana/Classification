import unittest
from lda import Dictionary
from lda import Entities
from lda import Document
from gensim import corpora

class testDictionary(unittest.TestCase):

    def setUp(self):
        self.doc = Document('TestDoc','Test to see if this text is added to dictionary.words')

        self.testDictionary = Dictionary()
        self.targetDictionary = Dictionary()

    def test_addDocument(self):
        document = Document()
        document.tokens = ['add', 'words', 'to', 'dictionary']
        document.specialCharacters = ['add' 'specialChars', '?!%$', 'add']

        dictionary = Dictionary()
        dictionary.addDocument(document)

        self.assertEqual(dictionary.specialCharacters, set(document.specialCharacters))
        self.assertEqual(set(dictionary.ids.values()), set(document.tokens))

        document2 = Document()
        document2.tokens = ['new', 'words']
        document2.specialCharacters = ['add', 'xx9']

        dictionary.addDocument(document2)

        document.specialCharacters.append('xx9')
        document.tokens.append('new')

        self.assertEqual(dictionary.specialCharacters, set(document.specialCharacters))
        self.assertEqual(set(dictionary.ids.values()), set(document.tokens))


    
    def test_getDictionaryId(self):
        self.targetDictionary.ids = {7:u'test', 11:u'if', 3:u'this', 1:u'set', 4:u'is', 6:u'converted', 5:u'to', 0:u'a', 2:u'dictionary', 8:u'representation', 9:u'corpus', 10:u'with'}
        self.assertEqual(9, self.targetDictionary.getDictionaryId('corpus'))
        self.assertEqual(7, self.targetDictionary.getDictionaryId('test'))

   
    def test_createEntities(self):
        testDictionary = Dictionary()
        collection = [Document('doc1','Test named entity recognition of a Collection of documents.'),Document('doc2',' African Commission is a named entity, also countries like Senegal and Lybia and names like Peter and Anna.'),Document('doc3', 'Also organizations like the United Nations or UNICEF should be recognized.')]
        testEntities = Entities('')
        testEntities.addEntities('ORGANIZATION', set([(u'african commission',1), (u'unicef', 1), (u'united nations', 1)]))
        testEntities.addEntities('PERSON', set([(u'anna',1), (u'peter',1)]))
        testEntities.addEntities('LOCATION', set([(u'senegal',1), (u'lybia',1)]))
        testDictionary.createEntities(collection)
        self.assertEqual(testEntities.__dict__, testDictionary.entities.__dict__)


    def test_invertDFS(self):
        self.testDictionary.ids.add_documents([['word', 'three', 'appears', 'twice'], ['once', 'three', 'appears', 'word'], ['three', 'twice']])
        inverseDFS = {1:['once'], 2:['twice', 'word', 'appears'], 3:['three']}
        self.testDictionary.invertDFS()
        self.assertEqual(inverseDFS, self.testDictionary.inverseDFS)



if __name__ =='__main__':
    unittest.main()
