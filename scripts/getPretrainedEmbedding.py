import numpy as np
import fastText

PATH = 'WordEmbedding/wiki.en.bin'
THRESHOLD = 40000

def getPretrainedEmbedding():

            word_embedding = fastText.load_model(PATH)

            initW = np.random.uniform(-0.25, 0.25, (THRESHOLD, 300))
            words = word_embedding.get_labels()[:THRESHOLD]
            indices = range(0,THRESHOLD)
            vocabulary = dict(zip(words, indices))

            for word,idx in vocabulary.iteritems():
                initW[idx] = word_embedding.get_word_vector(word)

            return initW, vocabulary




