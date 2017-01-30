#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lda import Collection, Entities, utils, Viewer 
from lda import dataframeUtils as df
import csv
import pandas
from gensim.parsing.preprocessing import STOPWORDS

def frequencyAnalysis():

    #### PARAMETERS ####
    evaluationFile = 'Documents/HRC_TopicAssignment.xlsx'
    
    keywords = pandas.read_excel(evaluationFile, 'Topics', header=None)
    keywords = utils.lowerList(list(keywords[0]))
    keywords.extend(['child', 'judicial system', 'independence'])

    assignedKeywords = pandas.read_excel(evaluationFile, 'Sheet1')

    path = "Documents/RightsDoc"
    filetype = "folder" 
    startDoc = 0
    docNumber = None
   
    filename = 'dataObjects/rightsDoc.txt'

    
    #### FREQUENCY ANALYSIS ####
    collection = Collection()
    collection.load(path, filetype, startDoc, docNumber)

    collection.undetectedKeywords = [] 
    collection.numberKeywords = 0

    for ind, doc in enumerate(collection.collection):
        
        doc.name = doc.title.replace('_', '/')[:-5]
        keywordFrequency = utils.countOccurance(doc.text, keywords)
        doc.entities.addEntities('KEYWORDS', utils.sortTupleList(keywordFrequency))

        doc.mostFrequent = doc.entities.getMostFrequent(5)
        mostFrequentWords = zip(*doc.mostFrequent)[0]

        targetKeywords = df.getRow(assignedKeywords, 'Symbol', doc.name, ['Topic 1', 'Topic 2', 'Topic 3'])
        targetKeywords = [keyword for keyword in targetKeywords if not str(keyword) =='nan'] 
        collection.numberKeywords += len(targetKeywords)

        doc.assignedKeywords = []
        
        for keyword in targetKeywords:
            isDetected = keyword.lower() in mostFrequentWords 
            if not isDetected:
                collection.undetectedKeywords.append((doc.name, keyword))
            doc.assignedKeywords.append((keyword.lower(), isDetected))

    collection.save(filename)

    html = Viewer()
    html.freqAnalysis(collection.collection, openHtml=False)
    html.freqAnalysis_eval(collection)

    with open('freqAnalysisOutput.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow([doc.name] + doc.mostFrequent) for doc in collection.collection]

      
if __name__ == "__main__":
    frequencyAnalysis()

