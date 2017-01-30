class Cluster:
    
    def __init__(self, number):
        self.number = number


    def createBinaryFeature(self, feature):
        self.createFeature(feature, True)
        self.createFeature(feature, False)
    
        
    def createFeature(self, feature, value):
        features = [doc for doc in self.documents if getattr(doc[0], feature)==value]
        setattr(self, feature + str(value), features)

