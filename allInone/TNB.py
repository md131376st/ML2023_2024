import numpy as np

from TCG import TCG

from Info import KFold


# tied Naive Bayes classifier
class TNB(TCG):
    def __init__(self, info=None):
        super().__init__(info)
        self.cov_classes_nbayes = []
        for i in range(self.classTypes):
            self.cov_classes_nbayes.append(
                self.cov_classes[i] * np.identity(self.info.data.shape[0]))
        self.cov_classes = self.cov_classes_nbayes
        pass


if __name__ == "__main__":
    KFold = KFold(5)
    for i in range(KFold.k):
        TiedNaiveBayes = TNB(KFold.infoSet[i])
        TiedNaiveBayes.applyTest()
        KFold.addscoreList(TiedNaiveBayes.checkAcc())
    KFold.ValidatClassfier("TNB")
