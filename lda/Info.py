class Info:

    def __init__(self):
        pass

    def setup(self):
        self.setIdentifier()
        self.setPath()
        self.setFileType()
        self.setCollectionName()
        self.setProcessedCollectionName()
        self.setModelPath()


    def setIdentifier(self):
        self.identifier = '%s_T%dP%dI%d' %(self.modelType, self.numberTopics, self.passes, self.iterations)
        if self.tfidf:
            self.identifier = self.identifier + '_tfidf'
        if self.whiteList != None:
            self.identifier = self.identifier + '_word2vec'


    def setFileType(self):
        if self.data == 'ICAAD':
            self.fileType = 'folder'
        elif self.data == 'NIPS':
            self.fileType = 'csv'
        elif self.data == 'scifibooks' or self.data =='HRC' or self.data == 'CRC' or self.data == 'Aleph':
            self.fileType = 'folder'
        else:
            print 'Data not found'


    def setPath(self):
        if self.data == 'ICAAD':
            self.path = 'Documents/ICAAD/files/txt'           
        elif self.data == 'NIPS':
            self.path = 'Documents/NIPS/Papers.csv'
        elif self.data == 'scifibooks':
            self.path = 'Documents/scifibookspdf'
        elif self.data == 'HRC':
            self.path = 'Documents/HRC/resolutions'
        elif self.data == 'CRC':
            self.path = 'Documents/CRC'
        elif self.data == 'Aleph':
            self.path = 'Documents/ALEPH'
        else:
            print 'Data not found'


    def setCollectionName(self):
        if self.includeEntities:
            self.collectionName = 'processedDocuments/'+self.data+'_entities'
        else:
            self.collectionName = 'processedDocuments/'+self.data+'_noEntities'
        if self.numberDoc:
            self.collectionName = self.collectionName + '_%d' % self.numberDoc
        
        if self.whiteList:
            self.collectionName = self.collectionName + '_word2vec'
        if self.removeNames:
            self.collectionName = self.collectionName + '_noNames'


    def setModelPath(self):
        self.modelPath = 'Models/' + self.data + '_' + self.identifier

    def setProcessedCollectionName(self):
        self.processedCollectionName = self.collectionName + '_' + self.identifier

    def saveToFile(self):
        dictionary = self.__dict__
        with open('html/' + self.data + '_' + self.identifier + '/info.txt', 'wb') as f:
            f.write('INFO - %s \n \n' % self.identifier)
            for key in dictionary:
                f.write(key + '  -  ' + str(dictionary[key]) +'\n')
        f.close()

