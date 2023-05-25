import numpy as np
import scipy.optimize
import scipy.special
from algorithemsBasic import AlgorithmBasic
from Info import KFold


# Binary logistic regression
class BLR(AlgorithmBasic):
    def __init__(self, info, l):
        super().__init__(info)
        self.l = l
        self.D = info.data.shape[0]  # dimensionality of features space
        self.K = len(set(info.label))  # number of classes
        self.N = info.data.shape[1]
        self.points, self.minvalue, self.d = scipy.optimize.fmin_l_bfgs_b(func=self.logreg_obj,
                                                                          x0=np.zeros(
                                                                              self.info.testData.shape[0] + 1),
                                                                          approx_grad=True,
                                                                          iprint=0)
        # print('Number of iterations: %s' % (self.d['funcalls']))

        pass

    def applyTest(self):
        w, b = self.points[0:-1], self.points[-1]
        testSize = self.info.testData.shape[1]
        self.score = np.zeros(testSize)
        for i in range(testSize):
            xi = self.info.testData[:, i:i + 1]
            s = np.dot(w.T, xi) + b
            self.score[i] = s

        pass

    def checkAcc(self):
        return self.info.testlable == (self.score > 0)
        # self.info.ValidatClassfier(sum(corrected_assigned_labels), classifier + " with lambda=" + str(self.l) + '')
        pass

    def __compute_zi(self, ci):
        return 2 * ci - 1

    def __compute_T(self):
        T = np.zeros(shape=(self.K, self.N))
        for i in range(self.N):
            label_xi = self.info.testlable[i]
            t = []
            for j in range(self.K):
                if j == label_xi:
                    t.append(1)
                else:
                    t.append(0)
            T[:, i] = t
        return T

    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        J = self.l / 2 * (np.linalg.norm(w) ** 2)
        summary = 0
        for i in range(self.info.testData.shape[1]):
            xi = self.info.testData[:, i:i + 1]
            ci = self.info.testlable[i]
            zi = self.__compute_zi(ci)
            summary += np.logaddexp(0, -zi * (np.dot(w.T, xi) + b))
        J += (1 / self.info.testData.shape[1]) * summary
        return J


if __name__ == "__main__":
    errorRate=[]
    lambdaList = [10 ** -6, 10 ** -3, 10 ** -1, 1, 10]
    for j in range(len(lambdaList)):
        KFold_ = KFold(5)
        for i in range(KFold_.k):
            # print("fold Number:" + str(i))
            logRegObj = BLR(KFold_.infoSet[i], lambdaList[j])
            logRegObj.applyTest()
            KFold_.addscoreList(logRegObj.checkAcc())
        KFold_.ValidatClassfier("BLR with lambda=" + str(lambdaList[j]) + '')
        errorRate.append(KFold_.err)
