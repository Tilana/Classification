from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
    
def tagEntities(text):
    nerTagger = StanfordNERTagger('stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','stanford-ner-2014-06-16/stanford-ner.jar', encoding='utf-8')
    return nerTagger.tag(word_tokenize(text))

def listEntities(taggedEntities):
    listEntities = []
    entity = []
    prevTag = "O"
    for token, tag in taggedEntities:
        if tag=="O":
            if entity:
                listEntities.append(entity)
            entity = []
        elif tag!="O" and tag==prevTag:
            entity.append((token, tag))
        elif tag!="O" and tag!=prevTag:
            if entity:
                listEntities.append(entity)
            entity = [(token,tag)]
        else:
            print "Error: Special case in listEntities detected"
        prevTag = tag
    return listEntities

def chunkEntities(listEntities):
    return [(u" ".join([token[0] for token in entity]), entity[0][1]) for entity in listEntities]

def getTokensWithTag(entitySet, tag):
    return (tag, [entity[0] for entity in entitySet if entity[1]==tag])

def orderNamedEntities(entitySet):
    return [getTokensWithTag(entitySet, 'ORGANIZATION'), getTokensWithTag(entitySet, 'LOCATION'), getTokensWithTag(entitySet, 'PERSON')]

def getNamedEntities(text):
    return orderNamedEntities(set(chunkEntities(listEntities(tagEntities(text)))))


