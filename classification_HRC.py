from lda import ClassificationModel, Viewer, Info

def classification_HRC():

    ##### PARAMETERS #####
    identifier = 'HRC_LDA_T15P985I1200_word2vec'
    path = 'html/%s/DocumentFeatures.csv' % identifier
    info = Info()
    info.data = 'HRC'
    info.identifier = 'classification'
    info.classifierType = 'DecisionTree'

    targetFeature = 'targetCategory1'
    droplist = ['File', 'Unnamed: 0']

    ### PREPROCESSING ###
    model = ClassificationModel(path, targetFeature, droplist, binary=False)
    
    ### SELECT TEST AND TRAINING DATA ###
    model.createTarget()
    model.dropFeatures()

    model.splitDataset(len(model.data)/2)

    ### CLASSIFICATION ###
    model.buildClassifier(info.classifierType)
    model.trainClassifier()
    model.predict()

    ### EVALUATION ###
    model.evaluate()
    model.evaluation.confusionMatrix()
    if not info.classifierType=='NeuralNet':
        model.computeFeatureImportance()

    model.getTaggedDocs()

    Viewer(info).classificationResults(model)

   
if __name__ == "__main__":
    classification_HRC()

