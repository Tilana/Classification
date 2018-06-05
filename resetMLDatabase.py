from pymongo import MongoClient

client  = MongoClient('localhost', 27017)
client.drop_database('machine_learning')
