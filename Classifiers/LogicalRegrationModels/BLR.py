import numpy as np
import scipy.optimize
import scipy.special
from matplotlib import pyplot as plt

from Classifiers.algorithemsBasic import AlgorithmBasic
from Data.Info import KFold


def vrow(x):
    return x.reshape((1, x.size))


# Binary logistic regression

def center_data(data, test):
    mean = np.mean(data, axis=1,keepdims=True)
    centered_data = data - mean
    centered_test = test - mean
    return centered_data, centered_test


def z_normalize_fuc(data, test):
    mean = np.mean(data, axis=1,keepdims=True)
    std = np.std(data, axis=1,keepdims=True)
    z_normalized_data = (data - mean) / std
    z_normalized_test = (test - mean) / std
    return z_normalized_data, z_normalized_test


class BLR(AlgorithmBasic):
    def __init__(self, info, l, pi_T=None, center=False, z_normalize=False, quadratic=False, slice=50):
        # if slice:
        #     info.data = info.data[:, ::50]
        #     info.label = info.label[::50]
        #     info.test = info.test[:, ::50]
        #     info.testData = info.testData[:, ::50]
        #     info.testlable = info.testlable[::50]

        super().__init__(info)
        if center:
            self.info.data, self.info.testData = center_data(self.info.data, self.info.testData)
        if z_normalize:
            self.info.data, self.info.testData = z_normalize_fuc(self.info.data, self.info.testData)
        if quadratic:
            self.info.data = self.add_quadratic_features(self.info.data)
            self.info.testData = self.add_quadratic_features(self.info.testData)
        self.l = l
        self.D = info.data.shape[0]  # dimensionality of features space
        self.K = len(set(info.label))  # number of classes
        self.N = info.data.shape[1]
        self.pi_T = pi_T
        self.score = None
        self.llr = None
        if pi_T == None:
            self.points = scipy.optimize.fmin_l_bfgs_b(
                func=self.logreg_obj,
                x0=np.zeros(
                    self.info.testData.shape[0] + 1)
            )[0]
        else:
            self.points = scipy.optimize.fmin_l_bfgs_b(
                    func=self.logreg_obj_weight,
                x0=np.zeros(
                    self.info.testData.shape[0] + 1)
            )[0]

        # print('Number of iterations: %s' % (self.d['funcalls']))

        pass

    def add_quadratic_features(self, data):
        # Create quadratic features
        quad_features = [data]
        for i in range(data.shape[0]):
            for j in range(i, data.shape[0]):
                quad_features.append(data[i] * data[j])
        return np.vstack(quad_features)

    def applyTest(self):
        w, b = self.points[:-1], self.points[-1]
        self.score = np.dot(w.T, self.info.testData) + b

        self.adjust_scores_to_llr()

        pass

    def checkAcc(self):
        return self.info.testlable == (self.score > 0)
        # self.info.ValidatClassfier(sum(corrected_assigned_labels), classifier + " with lambda=" + str(self.l) + '')
        pass

    def adjust_scores_to_llr(self):
        if self.pi_T == None:
            self.pi_T = (self.info.label == 1.0).sum() / self.N

        prior_log_odds = np.log(self.pi_T / (1 - self.pi_T))
        self.llr = self.score - prior_log_odds

    def compute_prior_weighted_log_likelihood(self, zi, linear_combination):

        pi_pos = self.pi_T
        pi_neg = 1 - self.pi_T

        log_likelihood_pos = pi_pos * np.logaddexp(0, -zi * linear_combination)
        log_likelihood_neg = pi_neg * np.logaddexp(0, zi * linear_combination)

        return log_likelihood_pos + log_likelihood_neg

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

    def logreg_obj_weight(self, v):
        w, b = v[:-1], v[-1]
        xi = self.info.data
        ci = self.info.label
        zi = 2 * ci - 1
        n_T = np.sum(ci == 1)
        n_F = np.sum(ci == 0)
        xi_weights = np.where(ci == 1, self.pi_T / n_T, (1 - self.pi_T) / n_F)
        linear_combination = np.dot(w.T, xi) + b  # Shape: (N,)
        log_likelihood = np.logaddexp(0, -zi * linear_combination)  # Shape: (N,)

        # Compute the gradient
        sigmoid = -zi / (1.0 + np.exp(zi * linear_combination)) * xi_weights  # Shape: (N,)
        J = self.l / 2 * (np.linalg.norm(w) ** 2) + np.sum(xi_weights * log_likelihood)

        gradient_w = (vrow(sigmoid) * xi).sum(1) + self.l * w.ravel()

        gradient_b = sigmoid.sum()  # Shape: scalar
        gradient = np.append(gradient_w, gradient_b)  # Shape: (D+1,)

        return J, gradient

    def logreg_obj(self, v):
        w, b = v[:-1], v[-1]
        xi = self.info.data
        ci = self.info.label
        zi = 2 * ci - 1
        linear_combination = np.dot(w.T, xi) + b  # Shape: (N,)
        log_likelihood = np.logaddexp(0, -zi * linear_combination)  # Shape: (N,)
        # Compute the gradient
        sigmoid = -zi / (1.0 + np.exp(zi * linear_combination))  # Shape: (N,)
        J = self.l / 2 * (np.linalg.norm(w) ** 2) + np.mean(log_likelihood)
        gradient_w = (vrow(sigmoid) * xi).mean(1) + self.l * w.ravel()

        gradient_b = sigmoid.mean()  # Shape: scalar

        # gradient_w = self.l * w + np.sum((xi_weights * sigmoid * zi)[:, np.newaxis] * xi.T, axis=0)  # Shape: (D,)
        # gradient_b = np.sum(xi_weights * sigmoid * zi)  # Shape: scalar

        gradient = np.append(gradient_w, gradient_b)  # Shape: (D+1,)

        return J, gradient


if __name__ == "__main__":
    errorRate = []
    lambdaList = np.logspace(-4, 2, 13)
    # lambdaList = [1e-3, 1e-1, 1.0]
    DCFs = []
    min_DCFs = []
    for j in range(len(lambdaList)):
        KFold_ = KFold(5, prior=0.1, pca=0)
        # for i in range(KFold_.k):
        # print("fold Number:" + str(i))
        logRegObj = BLR(KFold_.infoSet[0], lambdaList[j])
        logRegObj.applyTest()
        KFold_.addscoreList(logRegObj.checkAcc())
        KFold_.addLLR(logRegObj.llr)
        DCF, min_DCF = KFold_.ValidatClassfier("quadratic logistic regression  with lambda=" + str(lambdaList[j]) + '',
                                               fold_number=0, threshold=0.5)
        DCFs.append(DCF)
        min_DCFs.append(min_DCF)
        errorRate.append(KFold_.err)
    # plt.figure(figsize=(10, 7))
    # plt.xscale("log", base=10)
    # plt.plot(lambdaList, DCFs, label=f'DCF')
    # plt.plot(lambdaList, min_DCFs, '--', label=f'Min DCF')
    # plt.xlabel('$\lambda$')
    # plt.ylabel('Detection Cost Function (DCF)')
    # plt.title('DCF vs $\lambda$')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
