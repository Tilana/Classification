from gensim.models.wrappers import FastText
from gensim.models.keyedvectors import KeyedVectors
import pdb

PATH = '../WordEmbedding/FastText_wiki-news-300d-40000-subword.bin'
PATH = '../WordEmbedding/FastText_wiki-news-300d-40000-subword.vec'
PATH = '../WordEmbedding/FastText_wiki-news-300d-40000-subword.vec'

model = KeyedVectors.load_word2vec_format(PATH, binary=False)
model = FastText.load_fasttext_format(PATH)
model = FastText.load_word2vec_format(PATH)

pdb.set_trace()
