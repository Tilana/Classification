from lda import ClassificationModel

def cleanDataframe(data):
    dataWithText = data[data.text.notnull()]
    cleanDataframe = dataWithText.dropna(axis=1, how='all')
    cleanDataframe = cleanDataframe.reset_index()
    return cleanDataframe


def preprocessing(data, modelPath, vocabulary):

    data = cleanDataframe(data)
    model = ClassificationModel(data=data)
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        vecType = 'tfidf'
        model.buildPreprocessor(vecType=vecType, ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000, binary=False, vocabulary=vocabulary)
        model.trainPreprocessor(vecType)
        model.save(modelPath)



if __name__=='__main__':
    preprocessing()
