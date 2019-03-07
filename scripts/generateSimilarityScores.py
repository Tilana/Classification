""" Sentence Similarity Score Generator

converts multi-label classification data into sentence pairs with similarity scores
"""
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def themes_to_list(themes_string):
    themes_list = themes_string.split('\n- ')
    themes_list[0] = themes_list[0][2:]
    themes_list[-1] = themes_list[-1][:-1]
    return sorted(themes_list)


def generate_sample(data, state=1, N=100):
    sample = data[['Themes', 'Annotation']].sample(N, replace=True)
    col_names = {'Themes': 'Themes{}'.format(state), 'Annotation': 'sentence{}'.format(state)}
    sample.rename(columns=col_names, inplace=True)
    sample.reset_index(inplace=True)
    return sample


def concat_samples(sample1, sample2):
    sample = pd.concat([sample1, sample2], axis=1, ignore_index=False)
    sample = sample.drop(columns=['index'])
    return sample


def sample_from_category(data, category, state, N=100):
    subset = data[data['Themes_str'] == category]
    return generate_sample(subset, state, N)


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection)/float(len(union))
    return similarity


def similarity(themes1, themes2):
    return jaccard_similarity(set(themes1), set(themes2))


PATH = '../../data/OHCHR/uhri_rights.csv'

data = pd.read_csv(PATH)
data.dropna(subset=['Themes', 'Annotation'], inplace=True)

data['Themes'] = data.Themes.apply(themes_to_list)

N = 30000

sample1 = generate_sample(data, 1, N)
sample2 = generate_sample(data, 2, N)
sample = concat_samples(sample1, sample2)

data['Themes_str'] = data.Themes.str.join('; ')
counts = data['Themes_str'].value_counts()
frequent_themes = counts[(counts >= 50)].keys().tolist()

N = 50
for category in frequent_themes:
    sample1 = sample_from_category(data, category, 1, N)
    sample2 = sample_from_category(data, category, 2, N)
    category_sample = concat_samples(sample1, sample2)
    sample = sample.append(category_sample, ignore_index=True)

sample['similarity'] = sample.apply(lambda x: similarity(x.Themes1, x.Themes2), axis=1)
sample.hist(column='similarity')

plt.show()

high_sim = sample[sample.similarity >= 0.8]
low_sim = sample[sample.similarity <= 0.4]
rm_nr = len(low_sim) - len(high_sim)
rm_index = low_sim.sample(rm_nr).index
sample = sample.drop(rm_index)

sample.hist(column='similarity')
plt.show()

PROC_PATH = '../../data/OHCHR/uhri_rights_similarity_proc.pkl'
sample.to_csv(PROC_PATH, index=False)


