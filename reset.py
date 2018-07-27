from pymongo import MongoClient
import shutil
import sys
import os

model_dir = 'runs/'
ML_INSTANCES = ['ml_echr', 'ml_echr2']

def resetMLDatabase(instance):
    client  = MongoClient('localhost', 27017)
    client.drop_database(instance)

def removeCNNModels(instance):
    directory = os.path.join(model_dir, instance)
    shutil.rmtree(directory)
    oldmask = os.umask(000)
    os.makedirs(directory, 0777)
    os.umask(oldmask)



if __name__== "__main__":
    instance = sys.argv[1]
    if instance in ML_INSTANCES:
        resetMLDatabase(instance)
        removeCNNModels(instance)
        print 'RESET OF {} INSTANCE COMPLETED'.format(instance)
    else:
        print 'RESET FAILED: - GIVE VALID ML INSTANCE'



