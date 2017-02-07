import re

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
        return match


    def extractDigit(self, string):
        regex = r'\d+'
        match = re.search(regex, string)
        return int(match.group(0))


    





        
