import re
import pandas as pd

class FeatureExtractor:

    def __init__(self):
        print 'Build Feature Extractor'
        self.wordList = pd.read_csv('Documents/ICAAD/CategoryLists.csv')

    def year(self, title):
        regex = r'\[\d{4}\]'
        match = re.search(regex, title)
        year = self.extractDigit(match.group(0))
        return year 

    
    def court(self, title):
        regex = r'[A-Z]{4}'
        match = re.findall(regex, title)
        return match[0]


    def extractDigit(self, string):
        regex = r'\d+'
        match = re.search(regex, string)
        return int(match.group(0))

    
    def age(self, text):
        regex = r'\d+[-| ]*year\w*[-| ]*old|age of \d+|\d+ year\w* of age'
        return re.findall(regex, text)

    def ageRange(self, text):
        regex = r'((above|under|up to|between)( the age of \d+ (and \d+)*|\d* (and \d+)* years old))'
        results = re.findall(regex,text)
        return [elem[0] for elem in results]

    def minor(self, text):
        regex = r'under the age of \w+|younger than \w+'
        return re.findall(regex,text)

    def sentence(self, text):
        regex = '[\d+ to]* \d+ year\w* imprisonment'
        return re.findall(regex, text)

    def caseType(self, text):
        regex = r'RULING|JUDGEMENT|SUMMING UP|SENTENCE|JUDGMENT|DECISION|CHARGE|MINUTE|SENTENCING|REASONS|SUMMARY|APPEAL'
        result = re.findall(regex, text)
        return self.getFirstElement(result) 

    def victimRelated(self, text):
        regex = r'[\w ]+complainant|victim|plaintiff be \w+.+'
        return re.findall(regex, text)

    def accusedRelated(self, text):
        regex = r'[\w ]+accuse|you|perpetrator be [\w]*.'
        return re.findall(regex, text)

    def unique(self, l):
        return list(set(l))

    def getFirstElement(self, l):
        if l:
            return l[0]
        return None

    def findWordlistElem(self, text, col):
        wordlist = self.wordList[col].tolist()
        wordlist = [word for word in wordlist if type(word)==str]
        return re.findall('|'.join(wordlist), text)

    def groupTuples(self, tupleList):
        return [' '.join(elem) for elem in tupleList]


