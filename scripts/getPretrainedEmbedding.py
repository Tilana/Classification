import gensim.models.keyedvectors as w2v_model
import numpy as np
import tensorflow as tf

PATH = 'WordEmbedding/Word2Vec_GoogleNews-vectors-negative300.bin'

def getPretrainedEmbedding(vocab):

            word2vec = w2v_model.KeyedVectors.load_word2vec_format(PATH, binary=True)
            vocabulary = {key:value.index for key, value in word2vec.vocab.iteritems() if key.islower() and '_' not in key}

            vocabIntersection = list(set(vocab.keys()).intersection(vocabulary.keys()))
            wordsNotInWord2vec = list(set(vocab.keys()).difference(vocabIntersection))

            initW = np.random.uniform(-0.25, 0.25, (len(vocab), 300))
            for word in vocabIntersection:
                idx = vocab.get(word)
                initW[idx] = word2vec.word_vec(word)
            return initW


            # Memory efficient way?
            #with open(PATH, 'rb') as f:
            #    header = f.readline()
            #    vocab_size, layer_size = map(int, header.split())
            #    binary_len = np.dtype('float32').itemsize * layer_size

            #    for line in range(vocab_size):
            #        if line % 2000 == 0:
            #            print line
            #        word = []
            #        while True:
            #            ch = f.read(1).decode('latin-1')
            #            if ch == ' ':
            #                word = ''.join(word)
            #                break
            #            if ch !='\n':
            #                word.append(ch)
            #        idx = vocabulary.get(word)
            #        if idx != 0:
            #            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            #        else:
            #            f.read(binary_len)
            #return initW



if __name__=='__main__':
    getPretrainedEmbedding()





