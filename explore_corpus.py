import pandas as pd
from collections import Counter
import re
import random
import pdb

filename = '../data/echr.txt'
#filename = '../data/icaad.txt'
#filename = '../data/hrc.txt'
#filename = '../data/wiki+hr.txt'

keywords = ['rape', 'raped', 'sexual', 'assault', 'sexual assault']

with open(filename, 'r') as f:
    text = f.readline()

counter = Counter(text.split())

print('\n *** CORPUS: ****')
print('{}\n'.format(filename))

print('Number of Words: {}'.format(len(text.split())))
print('Vocabulary size: {} \n \n'.format(len(counter)))

for word in keywords:
    print('Frequency of {}: {} \n'.format(word, counter.get(word)))


def get_words_in_context(word):
    ind = [match.start() for match in re.finditer(word, text)]
    for i in range(10):
        rnd = random.randrange(len(ind))
        rnd_ind = ind[rnd]
        print('- {} \n'.format(text[rnd_ind - 50: rnd_ind + 50]))


get_words_in_context('raped')
