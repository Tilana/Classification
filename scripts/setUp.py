import lda.osHelper as osHelper
from lda import Info
import pandas as pd
import os


def setUp(categoryID, preprocessing=False):

    modelPath = osHelper.generateModelDirectory(categoryID)
    checkpointDir = os.path.join(modelPath, 'checkpoints')
    infoFile = os.path.join(modelPath, 'info.json')
    memoryFile = os.path.join(modelPath, 'memory.csv')

    osHelper.createFolderIfNotExistent(modelPath)
    osHelper.deleteFolderWithContent(checkpointDir)

    info = Info(infoFile)
    info.setup(categoryID, preprocessing)

    memory = pd.DataFrame(columns=['sentence', 'label', 'tokens', 'mapping', 'oov'])
    memory.to_csv(memoryFile, index=False)

    return True


if __name__=='__main__':
    setUp()
