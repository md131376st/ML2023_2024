import numpy as np
import scipy.special as ss

from algorithemsBasic import AlgorithmBasic
from Info import Info, KFold


class MGC(AlgorithmBasic):
    def __init__(self, info=None):
        super().__init__(info)


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
            for j in set(self.info.testlable):
                mu = self.mu_classes[int(j)]
                c = self.cov_classes[int(j)]
                score[int(j), :] = np.exp(self.logpdf_GAU_ND_1sample(xt, mu, c))
            self.score[:, i:i + 1] = score
        # we assume are prior probability is 1/2
        self.Sjoin = 1 / 2 * self.score
        self.logSJoint = np.log(self.Sjoin) + np.log(1 / 2)
        self.logSMarginal = ss.logsumexp(self.logSJoint, axis=0).reshape(1, -1)
        log_SPost = self.logSJoint - self.logSMarginal
        self.SPost = np.exp(log_SPost)
        pass

    def checkAcc(self):
        predicted_labels = np.argmax(self.SPost, axis=0)
        return self.info.testlable == predicted_labels
        pass

    def logpdf_GAU_ND_1sample(self, x, mu, C):
        M = x.shape[0]  # num of features of sample x
        mu = mu.reshape(M, 1)  # mean of the sample
        xc = x - mu  # x centered
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)


if __name__ == "__main__":
    KFold = KFold(3)
    for i in range(KFold.k):
        MGC_ = MGC(KFold.infoSet[i])
        MGC_.applyTest()
        KFold.addscoreList(MGC_.checkAcc())
    KFold.ValidatClassfier("MGC")
