import numpy as np
import scipy.optimize
import scipy.special
from Classifiers.algorithemsBasic import AlgorithmBasic
from Data.Info import KFold


# Binary logistic regression
class BLR(AlgorithmBasic):
    def __init__(self, info, l, pi_T=0.5):
        super().__init__(info)
        self.l = l
        self.D = info.data.shape[0]  # dimensionality of features space
        self.K = len(set(info.label))  # number of classes
        self.N = info.data.shape[1]
        self.pi_T = pi_T
        self.score = None
        self.llr = None
        self.points, self.minvalue, self.d = scipy.optimize.fmin_l_bfgs_b(
            func=self.logreg_obj,
            x0=np.zeros(
                self.info.testData.shape[0] + 1),
            approx_grad=False,
            iprint=0
        )

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

        self.adjust_scores_to_llr()

        pass

    def checkAcc(self):
        return self.info.testlable == (self.score > 0)
        # self.info.ValidatClassfier(sum(corrected_assigned_labels), classifier + " with lambda=" + str(self.l) + '')
        pass

    def adjust_scores_to_llr(self):
        prior_log_odds = np.log(self.pi_T / (1 - self.pi_T))
        self.llr = self.score - prior_log_odds

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
        # w, b = v[0:-1], v[-1]
        # J = self.l / 2 * (np.linalg.norm(w) ** 2)
        # summary = 0
        # for i in range(self.info.testData.shape[1]):
        #     xi = self.info.testData[:, i:i + 1]
        #     ci = self.info.testlable[i]
        #     zi = self.__compute_zi(ci)
        #     summary += np.logaddexp(0, -zi * (np.dot(w.T, xi) + b))
        # J += (1 / self.info.testData.shape[1]) * summary
        w, b = v[:-1], v[-1]
        xi = self.info.testData
        ci = self.info.testlable
        zi = 2 * ci - 1
        linear_combination = np.dot(w.T, xi) + b  # Shape: (N,)
        log_likelihood = np.logaddexp(0, -zi * linear_combination)  # Shape: (N,)

        # Compute the objective function value
        J = self.l / 2 * (np.linalg.norm(w) ** 2) + np.mean(log_likelihood)

        # Compute the gradient
        sigmoid = scipy.special.expit(-zi * linear_combination)  # Shape: (N,)
        gradient_w = -np.mean((sigmoid * zi) * xi, axis=1) + self.l * w  # Shape: (D,)
        gradient_b = -np.mean(sigmoid * zi)  # Shape: scalar

        gradient = np.append(gradient_w, gradient_b)  # Shape: (D+1,)

        return J, gradient


if __name__ == "__main__":
    errorRate = []
    lambdaList = [10 ** -6, 10 ** -3, 10 ** -1, 0, 1, 10]
    for j in range(len(lambdaList)):
        KFold_ = KFold(5, prior=0.5, pca=0)
        # for i in range(KFold_.k):
        # print("fold Number:" + str(i))
        logRegObj = BLR(KFold_.infoSet[0], lambdaList[j])
        logRegObj.applyTest()
        KFold_.addscoreList(logRegObj.checkAcc())
        KFold_.addLLR(logRegObj.llr)
        KFold_.ValidatClassfier("BLR with lambda=" + str(lambdaList[j]) + '', fold_number=0)
        errorRate.append(KFold_.err)
