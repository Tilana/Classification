import pandas as pd
import numpy as np
import sys
import pdb

def combineCols(df):
    evidences = zip(df.probability.tolist(), df['evidence.text'].tolist())
    evidences.sort(reverse=True)
    return evidences

def sortByProbability(df):
    df['max_probability'] = [elem[0][0] for elem in df.evidences.tolist()]
    df.sort_values(by='max_probability', ascending=True)
    return df

def averageProbability(evidences):
    probabilities = [elem[0] for elem in evidences]
    return np.average(probabilities)


def postProcessSuggestions(filename):

    PATH = '../'
    METADATA_PATH = PATH + 'metadata.json'

    data = pd.read_csv(PATH + filename)
    data = data[data.isEvidence.isnull()]
    data.drop(columns=['isEvidence'], inplace=True)
    data.drop_duplicates(inplace=True)

    docs = data.groupby(['value','document'])

    output = pd.DataFrame()
    output['nr_evidences'] = docs['evidence.text'].count()
    output['evidences'] = docs.apply(combineCols)
    output.reset_index(inplace=True)

    metadata = pd.read_json(METADATA_PATH)
    combined = output.merge(metadata, left_on='document', right_on='sharedId')
    combined.drop(columns=['file', 'sharedId'], inplace=True)

    combined['avg_probability'] = combined.evidences.apply(averageProbability)
    combined = combined.reindex(columns=['document', 'title', 'nr_evidences', 'avg_probability', 'evidences'])
    combined.sort_values(['nr_evidences', 'avg_probability'], ascending=[False, False], inplace=True)
    combined.to_csv(PATH + 'proc_' + filename, encoding='utf8', index=False)

if __name__=='__main__':
    filename = sys.argv[1]
    postProcessSuggestions(filename)


