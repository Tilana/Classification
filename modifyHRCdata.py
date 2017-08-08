from lda import Collection
import numpy as np
import pdb
import csv


def modifyHRCdata():

    path = 'processedData/RightDocs'
    collection = Collection().load(path) 
   
    listedTerms = collection.data.topics.unique()
    listedTerms = [elem.encode('utf8') for elem in listedTerms[1:]]

    topics = set()
    for term in listedTerms:
            for topic in term.split(','):
                topics.add(topic)

    for topic in topics:
        collection.data[topic] = 0
                

    for index, row in collection.data.iterrows():
        try:
            if not str(row['topics'])=='nan':
                for topic in row.topics.split(','):
                    collection.data.loc[index, topic] = 1
        except:
            pass

    collection.save(path+'_topics')
    
    with open('Documents/HRC_topics.csv', 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        for item in topics:
            wr.writerow([item,])
        #wr.writerows(topics)

    pdb.set_trace()




if __name__ == '__main__':
    modifyHRCdata()


