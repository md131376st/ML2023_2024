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
    pca_list = [0, 2, 3, 4, 5]
    dcf = []
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.5, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = NBG(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(KFold_.ValidatClassfier(f"Naive Bayes Gaussian prior:0.5  cfn:1, cfp:1 pca:{pca} ", fold_number=0))
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.9, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = NBG(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(KFold_.ValidatClassfier(f"Naive Bayes Gaussian prior:0.9  cfn:1, cfp:1 pca:{pca}", fold_number=0))
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.1, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = NBG(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(KFold_.ValidatClassfier(f"Naive Bayes Gaussian prior:0.1  cfn:1, cfp:1 pca:{pca}", fold_number=0))
    print("Min DCF " + str(min(dcf)))
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, cfn=1, cfp=9, pca=pca)
    #     # for i in range(KFold.k):
    #     MGC_ = NBG(KFold_.infoSet[0], KFold_.pi)
    #     MGC_.applyTest()
    #     KFold_.addscoreList(MGC_.checkAcc())
    #     KFold_.addLLR(MGC_.foldLLR)
    #     dcf.append(KFold_.ValidatClassfier(f"Naive Bayes Gaussian prior:0.5  cfn:1, cfp:9 pca:{pca}", fold_number=0))
    #
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, cfn=9, cfp=1, pca=pca)
    #     # for i in range(KFold.k):
    #     MGC_ = NBG(KFold_.infoSet[0], KFold_.pi)
    #     MGC_.applyTest()
    #     KFold_.addscoreList(MGC_.checkAcc())
    #     KFold_.addLLR(MGC_.foldLLR)
    #     dcf.append(KFold_.ValidatClassfier(f"Naive Bayes Gaussian prior:0.5  cfn:9, cfp:1 pca:{pca}", fold_number=0))
    # print("Min normalize DCF " + str(min(dcf)))



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
