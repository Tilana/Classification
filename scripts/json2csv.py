import pandas as pd
import pdb

PATH = '../data/ECHR/suggestions_data_protection_cnn.json'
METADATA_PATH = '../data/ECHR/metadata.json'

data = pd.read_json(PATH)
metadata = pd.read_json(METADATA_PATH)

metadata['file'] = metadata.file.apply(lambda x: x.get('filename'))

data['evidence'] = data.evidence.apply(lambda x: x.get('text'))
data.dropna(subset=['probability'], inplace=True)
data.drop_duplicates(inplace=True)

docs = data.groupby('document')
#docs['nr_evidences'] = len(docs.evidence)

def combineCols(df):
    evidences = zip(df.probability.tolist(), df.evidence.tolist())
    evidences.sort(reverse=True)
    return evidences

def sortByProbability(df):
    df['max_probability'] = [elem[0][0] for elem in df.evidences.tolist()]
    df.sort_values(by='max_probability', ascending=True)
    return df

output = pd.DataFrame()
output['nr_evidences'] = docs.evidence.count()
output['evidences'] = docs.apply(combineCols) #, axis=1)
output.reset_index(inplace=True)

combined = output.merge(metadata, left_on='document', right_on='sharedId')
combined.drop(columns=['file', 'sharedId'], inplace=True)
combined = combined.reindex(columns=['document', 'title', 'nr_evidences', 'evidences'])

combined['max_probability'] = [evidence[0][0] for evidence in combined.evidences.tolist()]
combined.sort_values(['nr_evidences', 'max_probability'], ascending=[False, False], inplace=True)
combined.to_csv('../data/ECHR_data_protection_cnn.csv', encoding='utf8')

