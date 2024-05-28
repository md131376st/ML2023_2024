import numpy as np

from Classifiers.GenerativeModels.TCG import TCG

from Data.Info import KFold


# tied Naive Bayes classifier
class TNB(TCG):
    def __init__(self, info=None, prior=1 / 2):
        super().__init__(info, prior)
        self.cov_classes_nbayes = []
        for i in range(self.classTypes):
            self.cov_classes_nbayes.append(
                self.cov_classes[i] * np.identity(self.info.data.shape[0]))
        self.cov_classes = self.cov_classes_nbayes
        pass


if __name__ == "__main__":
    pca_list = [0, 2, 3, 4, 5]
    dcf = []
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.5, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = TNB(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(
            KFold_.ValidatClassfier(f"Tied Naive Bayes Gaussian prior:0.5  cfn:1, cfp:1 pca:{pca} ", fold_number=0))
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.9, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = TNB(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(
            KFold_.ValidatClassfier(f"Tied Naive Bayes Gaussian prior:0.9  cfn:1, cfp:1 pca:{pca}", fold_number=0))
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.1, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = TNB(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(
            KFold_.ValidatClassfier(f"Tied Naive Bayes Gaussian prior:0.1  cfn:1, cfp:1 pca:{pca}", fold_number=0))
    print("Min DCF " + str(min(dcf)))
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, cfn=1, cfp=9, pca=pca)
    #     # for i in range(KFold.k):
    #     MGC_ = TNB(KFold_.infoSet[0], KFold_.pi)
    #     MGC_.applyTest()
    #     KFold_.addscoreList(MGC_.checkAcc())
    #     KFold_.addLLR(MGC_.foldLLR)
    #     dcf.append(KFold_.ValidatClassfier(f"Tied Naive Bayes Gaussian prior:0.5  cfn:1, cfp:9 pca:{pca}", fold_number=0))
    #
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, cfn=9, cfp=1, pca=pca)
    #     # for i in range(KFold.k):
    #     MGC_ = TNB(KFold_.infoSet[0], KFold_.pi)
    #     MGC_.applyTest()
    #     KFold_.addscoreList(MGC_.checkAcc())
    #     KFold_.addLLR(MGC_.foldLLR)
    #     dcf.append(KFold_.ValidatClassfier(f"Tied Naive Bayes Gaussian prior:0.5  cfn:9, cfp:1 pca:{pca}", fold_number=0))
    # print("Min normalize DCF " + str(min(dcf)))
    # pca_list = [2, 3, 4, 5]
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, pca=pca)
    #
    # # for i in range(KFold.k):
    #     TiedNaiveBayes = TNB(KFold_.infoSet[0])
    #     TiedNaiveBayes.applyTest()
    #     KFold_.addscoreList(TiedNaiveBayes.checkAcc())
    #     KFold_.addLLR(TiedNaiveBayes.foldLLR)
    #     KFold_.ValidatClassfier("Tied Naive Bayes Gaussian", fold_number=0)
