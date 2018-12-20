import pandas as pd
from lda.Viewer import Viewer
import pdb

MODEL1 = 'keyword_rape_fasttext_hr_dim200_20_window'
MODEL2 = 'keyword_rape_fasttext_hr_stem+unstem_dim200_window'

PATH1 = './{}.pkl'.format(MODEL1)
PATH2 = './{}.pkl'.format(MODEL2)

data1 = pd.read_pickle(PATH1)
data1.drop(columns=['proc_sentence', 'predictedLabel'], axis=1, inplace=True)

data2 = pd.read_pickle(PATH2)
data2.drop(columns=['proc_sentence', 'predictedLabel'], axis=1, inplace=True)

data1['similarity'] = data1['similarity'] * 100
data1['similarity'] = data1['similarity'].round(2)
data2['similarity'] = data2['similarity'] * 100
data2['similarity'] = data2['similarity'].round(2)

diff = data1[~(data1 == data2).tags]
diff['data2_tags'] = data2.tags
diff['data2_similarity'] = data2.similarity

diff['sim_diff'] = data1['similarity'] - data2['similarity']
diff['abs_diff'] = abs(diff['sim_diff'])

diff.rename(index=str,
            columns={'tags': 'data1_tags', 'similarity': 'data1_similarity'},
            inplace=True)
diff.drop(columns=['id', 'targetLabel', 'sim_diff'], inplace=True)
diff.sort_values('abs_diff', inplace=True, ascending=False)

tags = ['data1_tags', 'data2_tags']
overview = diff[tags].groupby(tags).size()
overview = pd.DataFrame(overview)

pd.set_option('display.max_colwidth', 500)
viewer = Viewer('compare')
viewer.compareDataFrames('test', MODEL1, MODEL2, overview, diff)


