from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import fastText
import pdb


model_name = 'hr'

path = '../fastText/{}.bin'.format(model_name)
model = fastText.load_model(path)

vocab = model.get_words()
word = 'sexual_assault'
if word in vocab:
    print('word {} in vocab'.format(word))


def average_subwords(word, plot=0):
    subwords = model.get_subwords(word)[0]
    representations = []
    for subword in subwords:
        representations.append(model.get_word_vector(subword))
    matrix = np.array(representations)
    if plot:
        df = pd.DataFrame(np.swapaxes(matrix, 0, 1), columns=subwords)
        plot_heatmap(df)
    return np.average(matrix, axis=0)


def plot_heatmap(df, num=1):
    f, ax = plt.subplots(num)
    sns.heatmap(df, ax=ax)
    plt.show()


def query_word(word, matrix, model):
    idx = int(model.get_word_id(word))
    return matrix[idx]



avg_vec = average_subwords(word, 1)
vec = model.get_word_vector(word)

matrix = model.get_output_matrix()
pca = decomposition.PCA(n_components=5)
x_std = StandardScaler().fit_transform(matrix)
pca_matrix = pca.fit_transform(x_std)

keywords = ['rape', 'raped', 'stepfather', 'robbery']
for ind, mat in enumerate([matrix, pca_matrix]):
    representations = [query_word(word, mat, model) for word in keywords]
    df = pd.DataFrame(np.swapaxes(np.array(representations), 0, 1), columns=keywords)
    plot_heatmap(df)



print(avg_vec[:10])
print(vec[:10])



pdb.set_trace()
