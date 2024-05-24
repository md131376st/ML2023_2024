import numpy as np

from Classifiers.GenerativeModels.TCG import TCG

from Data.Info import KFold


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
    pca_list = [2, 3, 4, 5]
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.5, pca=pca)

    # for i in range(KFold.k):
        TiedNaiveBayes = TNB(KFold_.infoSet[0])
        TiedNaiveBayes.applyTest()
        KFold_.addscoreList(TiedNaiveBayes.checkAcc())
        KFold_.addLLR(TiedNaiveBayes.foldLLR)
        KFold_.ValidatClassfier("Tied Naive Bayes Gaussian", fold_number=0)
