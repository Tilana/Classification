import pandas as pd

splitColumns = lambda x: pd.Series([elem for elem in x.split(',')])

def getLabel(labelList):
    for label in labelList.split(','):
        if label in LABELS:
            return label

filename = 'UPR_DATABASE.csv'
path = '../data/UPR/'

LABELS = ['Enforced disappearances', 'Disabilities', 'Freedom of opinion and expression', 'Human rights violations by state agents', 'International humanitarian law', 'Migrants', 'Minorities', 'Poverty', 'Trafficking']

data = pd.read_csv(path + filename)

#multiLabels = data.Issue.apply(splitColumns)
data['label'] = data.Issue.apply(getLabel)

data.to_csv(path+filename)

