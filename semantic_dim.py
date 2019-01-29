import numpy as np
import fastText
import pdb

we_model = 'wiki+hr'
we_model = 'hr'

W2V_PATH = '../fastText/{}.bin'.format(we_model)
model = fastText.load_model(W2V_PATH)

evidence = 'rape'
domain = ['rape', 'incest', 'sexual', 'assault']
words = ['raped', 'assault', 'murder', 'court', 'judge', 'plaintiff', 'guilty', 'apple', 'girl']
DIM = 5


def get_semantic_dimensions(model, evidence):
    evidence_embedding = model.get_word_vector(evidence)
    avg_embedding = np.average(model.get_output_matrix(), axis=0)

    diff = evidence_embedding - avg_embedding
    semantic_dim = np.argsort(abs(diff))[-DIM:]
    return semantic_dim


def get_pos_neg_dim(embedding, semantic_dim):
    neg_dim = []
    pos_dim = []
    for ind in semantic_dim:
        value = embedding[ind]
        if value < 0:
            neg_dim.append(ind)
        else:
            pos_dim.append(ind)
    return (pos_dim, neg_dim)


evd_dim = get_semantic_dimensions(model, evidence)
for word in words:
    word_dim = get_semantic_dimensions(model, word)
    intersection = set(evd_dim).intersection(set(word_dim))
    print(word)
    print(word_dim)
    print('Number of matches: {}'.format(len(intersection)))
    print('Matching Dim: {}'.format(list(intersection)))
    print('----\n')

pdb.set_trace()
