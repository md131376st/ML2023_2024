import numpy as np
import scipy.special as ss

from Classifiers.algorithemsBasic import AlgorithmBasic
from Data.Info import KFold


class MGC(AlgorithmBasic):
    def __init__(self, info=None, prior=1 / 2):
        super().__init__(info, prior)

        self.classTypes = len(set(self.info.testlable))
        self.mu_classes = []  # list of empirical mean for each class
        self.cov_classes = []  # list of covariance matrix for each class
        # calculate empirical mean and covariance for each class
        for i in set(self.info.label):
            trainData = self.info.data[:, self.info.label == i]
            size = trainData.shape[1]
            mean = trainData.mean(axis=1).reshape(-1, 1)
            covariance = 1 / size * np.dot(trainData - mean, (trainData - mean).T)
            self.mu_classes.append(mean)
            self.cov_classes.append(covariance)

    def applyTest(self):
        self.score = np.zeros(shape=(self.classTypes, self.info.testData.shape[1]))
        for i in range(self.info.testData.shape[1]):
            xt = self.info.testData[:, i:i + 1]
            score = np.zeros(shape=(self.classTypes, 1))
            for j in range(len(set(self.info.testlable))):
                mu = self.mu_classes[int(j)]
                c = self.cov_classes[int(j)]
                score[int(j), :] = self.logpdf_GAU_ND_1sample(xt, mu, c) + np.log(self.prior)
            self.score[:, i:i + 1] = score
        self.logSMarginal = ss.logsumexp(self.score, axis=0).reshape(1, -1)
        log_SPost = self.score - self.logSMarginal
        self.SPost = np.exp(log_SPost)
        self.Destination()
        pass

    def checkAcc(self):
        predicted_labels = np.argmax(self.SPost, axis=0)
        return self.info.testlable == predicted_labels
        pass

    def Destination(self):
        self.foldLLR = np.zeros(self.info.testData.shape[1])
        for j in range(self.info.testData.shape[1]):
            self.foldLLR[j] = self.SPost[1][j] - self.SPost[0][j]

    def logpdf_GAU_ND_1sample(self, x, mu, C):
        M = x.shape[0]  # num of features of sample x
        mu = mu.reshape(M, 1)  # mean of the sample
        xc = x - mu  # x centered
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)


if __name__ == "__main__":
    pca_list = [0, 2, 3, 4, 5]
    dcf = []
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.5, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = MGC(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(KFold_.ValidatClassfier(f"MVG prior:0.5  cfn:1, cfp:1 pca:{pca} ", fold_number=0))
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.9, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = MGC(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(KFold_.ValidatClassfier(f"MVG prior:0.9  cfn:1, cfp:1 pca:{pca}", fold_number=0))
    for pca in pca_list:
        KFold_ = KFold(5, prior=0.1, cfn=1, cfp=1, pca=pca)
        # for i in range(KFold.k):
        MGC_ = MGC(KFold_.infoSet[0], KFold_.pi)
        MGC_.applyTest()
        KFold_.addscoreList(MGC_.checkAcc())
        KFold_.addLLR(MGC_.foldLLR)
        dcf.append(KFold_.ValidatClassfier(f"MVG prior:0.1  cfn:1, cfp:1 pca:{pca}", fold_number=0))
    print("Min DCF " + str(min(dcf)))
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, cfn=1, cfp=9, pca=pca)
    #     # for i in range(KFold.k):
    #     MGC_ = MGC(KFold_.infoSet[0], KFold_.pi)
    #     MGC_.applyTest()
    #     KFold_.addscoreList(MGC_.checkAcc())
    #     KFold_.addLLR(MGC_.foldLLR)
    #     dcf.append(KFold_.ValidatClassfier(f"MVG prior:0.5  cfn:1, cfp:9 pca:{pca}", fold_number=0))
    #
    # for pca in pca_list:
    #     KFold_ = KFold(5, prior=0.5, cfn=9, cfp=1, pca=pca)
    #     # for i in range(KFold.k):
    #     MGC_ = MGC(KFold_.infoSet[0], KFold_.pi)
    #     MGC_.applyTest()
    #     KFold_.addscoreList(MGC_.checkAcc())
    #     KFold_.addLLR(MGC_.foldLLR)
    #     dcf.append(KFold_.ValidatClassfier(f"MVG prior:0.5  cfn:9, cfp:1 pca:{pca}", fold_number=0))
    # print("Min normalize DCF " + str(min(dcf)))

    # KFold = KFold(10,prior= 0.9,pca=11)
    # for i in range(KFold.k):
    #     MGC_ = MGC(KFold.infoSet[i], KFold.pi)
    #     MGC_.applyTest()
    #     hi = MGC_.checkAcc()
    #     KFold.addscoreList(MGC_.checkAcc())
    #     KFold.addLLR(MGC_.foldLLR)
    # KFold.ValidatClassfier("MGC",1)

    # KFold = KFold(10,prior= 0.1,pca=8)
    # for i in range(KFold.k):
    #     MGC_ = MGC(KFold.infoSet[i], KFold.pi)
    #     MGC_.applyTest()
    #     hi = MGC_.checkAcc()
    #     KFold.addscoreList(MGC_.checkAcc())
    #     KFold.addLLR(MGC_.foldLLR)
    # KFold.ValidatClassfier("MGC",1)
