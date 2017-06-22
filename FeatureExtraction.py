from lda import FeatureExtractor, Preprocessor
import pdb

def applyToRows(data, field, fun, name, args=None):
    if args:
        data[name] = data.apply(lambda doc: fun(doc[field], args), axis=1)
    else:
        data[name] = data.apply(lambda doc: fun(doc[field]), axis=1)



def FeatureExtraction(data):

    preprocessor = Preprocessor()
    #pdb.set_trace()
    applyToRows(data, 'text', preprocessor.cleanText, 'cleanText') 
    applyToRows(data, 'cleanText', preprocessor.numbersInTextToDigits, 'cleanText') 
    
    extractor = FeatureExtractor()
    applyToRows(data, 'title', extractor.court, 'ext_Court') 
    applyToRows(data, 'title', extractor.year, 'ext_Year') 
    applyToRows(data, 'cleanText', extractor.age, 'ext_Age') 
    applyToRows(data, 'cleanText', extractor.ageRange, 'ext_AgeRange') 
    applyToRows(data, 'cleanText', extractor.sentence, 'ext_Sentences') 
    applyToRows(data, 'text', extractor.caseType, 'ext_CaseType') 
    applyToRows(data, 'text', extractor.entities, 'entities')

    applyToRows(data, 'cleanText', extractor.findWordlistElem, 'ext_reconciliation', 'reconciliation')
    applyToRows(data, 'cleanText', extractor.findWordlistElem, 'ext_FamilyRelations', 'family')
    applyToRows(data, 'cleanText', extractor.findWordlistElem, 'ext_sentencingType', 'sentencing')
    
    return data


if __name__ == '__main__':
    FeatureExtraction()
