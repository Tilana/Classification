from nltk import sent_tokenize
import pandas as pd

PATH = '../../data/sampleTexts/'

data = pd.read_csv(PATH + 'sample.csv', encoding='utf8')

for ind,doc in data.iterrows():

    sentences = sent_tokenize(doc.text)
    sentences = [(sentence + '\n\n').encode('utf8') for sentence in sentences]

    with open(PATH + doc['_id'] + '.txt', 'wb') as f:
        f.writelines(sentences)

    f.close()



