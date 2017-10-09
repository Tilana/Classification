import os
import glob
import pdb

def deleteFolderContent(path):
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)

