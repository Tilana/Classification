import pdb
import json
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
database = client.uwazi_development
collection = database.entities

for document in collection.find():
    print document['title']

print 'Number of Documents: ' + str(collection.count())

collection.distinct('type')
documentData = collection.find({'type': 'document'})

textData = collection.find({'fullText': {"$exists": True}})
print 'Number of Text Documents ' + str(textData.count())

for document in textData: 
    print document['title']

print document.keys()
print document['metadata']


pdb.set_trace()

