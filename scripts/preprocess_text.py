from nltk import PorterStemmer
import pdb

PATH = '../../data/wiki+hr'
PATH = '../../data/hr'

def findSubstrings(tokens, key):
    relWords = [token for token in tokens if key==token]
    return relWords

with open(PATH + '.txt', 'r') as f:
    text = f.read()

tokens = text.split()
rape_substrings = findSubstrings(tokens, 'rape')
raped_substrings = findSubstrings(tokens, 'raped')
rapes_substrings = findSubstrings(tokens, 'rapes')

pdb.set_trace()

ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in tokens]

with open(PATH + '_stemmed.txt', 'w') as f:
    f.write(' '.join(stemmed_tokens))

pdb.set_trace()

