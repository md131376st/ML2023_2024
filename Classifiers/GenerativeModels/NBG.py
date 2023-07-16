import numpy as np
from Classifiers.GenerativeModels.MGC import MGC
from Data.Info import KFold


# calculate Naive Bayes Gaussian Classifier
class NBG(MGC):
    def __init__(self, info, prior=1 / 2):
        super().__init__(info, prior)
        self.cov_classes_nbayes = []
        # calculate diagonal covariance class
        for i in range(self.classTypes):
            self.cov_classes_nbayes.append(
                self.cov_classes[i] * np.identity(self.info.data.shape[0]))
        self.cov_classes = self.cov_classes_nbayes


if __name__ == "__main__":
    KFold = KFold(10,prior= 0.1,pca=0)
    for i in range(KFold.k):
        NaiveBayes = NBG(KFold.infoSet[i])
        NaiveBayes.applyTest()
        KFold.addscoreList(NaiveBayes.checkAcc())
        KFold.addLLR(NaiveBayes.foldLLR)
    KFold.ValidatClassfier("NBG",1)
    #
    # KFold = KFold(10,prior= 0.1,pca=11)
    # for i in range(KFold.k):
    #     NaiveBayes = NBG(KFold.infoSet[i])
    #     NaiveBayes.applyTest()
    #     KFold.addscoreList(NaiveBayes.checkAcc())
    #     KFold.addLLR(NaiveBayes.foldLLR)
    # KFold.ValidatClassfier("NBG",1)
    #
    # KFold = KFold(10,prior= 0.5,pca=8)
    # for i in range(KFold.k):
    #     NaiveBayes = NBG(KFold.infoSet[i])
    #     NaiveBayes.applyTest()
    #     KFold.addscoreList(NaiveBayes.checkAcc())
    #     KFold.addLLR(NaiveBayes.foldLLR)
    # KFold.ValidatClassfier("NBG",1)
