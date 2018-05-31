import os
import glob

def deleteFolderContent(path):
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)

def deleteFolderWithContent(path):
    try:
        deleteFolderContent(path)
        os.rmdir(path)
    except:
        pass


def createFolderIfNotExistent(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generateModelDirectory(category):
    return os.path.join('runs', category)
