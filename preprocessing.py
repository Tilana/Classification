from lda import ClassificationModel
import pdb

def preprocessing(data, modelPath):

    model = ClassificationModel(data=data)
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(vecType='tfidf', ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000, binary=True)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)



if __name__=='__main__':
    preprocessing()
