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
    KFold = KFold(10)
    for i in range(KFold.k):
        TCG_ = TCG(KFold.infoSet[i])
        TCG_.applyTest()
        KFold.addscoreList(TCG_.checkAcc())
    KFold.ValidatClassfier("TCG")
