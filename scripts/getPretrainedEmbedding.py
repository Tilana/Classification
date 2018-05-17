import gensim.models.keyedvectors as w2v_model, numpy as np, tensorflow as tf
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


if __name__ == '__main__':
    getPretrainedEmbedding()
