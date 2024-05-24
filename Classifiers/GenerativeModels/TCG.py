from Classifiers.GenerativeModels.MGC import MGC
from Data.Info import KFold


# tied covariance Gaussian classifier
class TCG(MGC):
    def __init__(self, info):
        super().__init__(info)
        self.num_samples_per_class = [sum(self.info.label == i) for i in range(self.classTypes)]
        self.tied_cov = 0
        for i in range(self.classTypes):
            self.tied_cov += (self.num_samples_per_class[i] * self.cov_classes[i])
        self.tied_cov *= 1 / sum(self.num_samples_per_class)
        for i in range(len(self.cov_classes)):
            self.cov_classes[i] = self.tied_cov


if __name__ == "__main__":
    pca_list = [2, 3, 4, 5]
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.5, pca=pca)
    # for i in range(KFold.k):
        TCG_ = TCG(KFold_.infoSet[0])
        TCG_.applyTest()
        KFold_.addscoreList(TCG_.checkAcc())
        KFold_.addLLR(TCG_.foldLLR)
        KFold_.ValidatClassfier("tied Gaussian", fold_number=0)

