import re
import text2num

class FeatureExtractor:

    def __init__(self):
        print 'Build Feature Extractor'
        pass


    def extractYear(self, title):
        regex = r'\[\d{4}\]'
        match = re.search(regex, title)
        year = self.extractDigit(match.group(0))
        return year 

    
    def extractCourt(self, title):
        regex = r'[A-Z]{4}'
        match = re.findall(regex, title)
        return match[0]


    def extractDigit(self, string):
        regex = r'\d+'
        match = re.search(regex, string)
        return int(match.group(0))

    
    def extractAge(self, text):
        regex = r'\w+[-| ]*year\w*[-| ]*old|age of \w+|\w+ year\w* of age'
        return re.findall(regex, text)

    def extractMinor(self, text):
        regex = r'under the age of \w+|younger than \w+'
        return re.findall(regex,text)

    def extracSentence(self, text):
        regex = '\w+ year\w* imprisonment'
        return re.findall(regex, text)



    





        
