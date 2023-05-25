import numpy as np
from MGC import MGC
from Info import KFold



# calculate Naive Bayes Gaussian Classifier
class NBG(MGC):
    def __init__(self, info):
        super().__init__(info)
        self.cov_classes_nbayes = []
        # calculate diagonal covariance class
        for i in range(self.classTypes):
            self.cov_classes_nbayes.append(
                self.cov_classes[i] * np.identity(self.info.data.shape[0]))
        self.cov_classes = self.cov_classes_nbayes


if __name__ == "__main__":
    KFold = KFold(5)
    for i in range(KFold.k):
        NaiveBayes = NBG(KFold.infoSet[i])
        NaiveBayes.applyTest()
        KFold.addscoreList(NaiveBayes.checkAcc())
    KFold.ValidatClassfier("NBG")
