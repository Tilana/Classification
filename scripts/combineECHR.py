import pandas as pd
import pdb
import json

id_file = '../data/ECHR/echr_ids.json'
data_file = '../data/ECHR/ECHR-GC.pkl'


def getID(filename):
    return filename['originalname'].split('.')[0]

def setID(filename):
    return '-'.join(filename.split('-')[0:2])



data = pd.read_pickle(data_file)
data.rename(columns={'title':'filename'}, inplace=True)
data['id'] = data.filename.apply(setID)

ids = pd.read_json(id_file)
ids['id'] = ids.file.apply(getID)

full_data = pd.merge(data, ids, on='id', how='inner')
full_data.drop(columns=['file', 'id'], inplace=True)
full_data.to_csv('../data/ECHR/echr.csv', encoding='utf8')


