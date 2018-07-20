from pymongo import MongoClient
import shutil
import os

model_dir = 'runs/'

def resetMLDatabase():
    client  = MongoClient('localhost', 27017)
    client.drop_database('machine_learning')

def removeCNNModels():
    shutil.rmtree(model_dir)
    oldmask = os.umask(000)
    os.makedirs(model_dir, 0777)
    os.umask(oldmask)



if __name__== "__main__":
    resetMLDatabase()
    removeCNNModels()



