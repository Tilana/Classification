from lda import FeatureExtractor, Preprocessor
import pdb


def FeatureExtraction_ICAAD(data):

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
    FeatureExtraction_ICAAD()
