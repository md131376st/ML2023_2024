import numpy as np
from matplotlib import pyplot as plt

from Classifiers.algorithemsBasic import AlgorithmBasic
from Data.Info import KFold
import scipy.optimize
import scipy.special


def center_data(data, test):
    mean = np.mean(data, axis=1, keepdims=True)
    centered_data = data - mean
    centered_test = test - mean
    return centered_data, centered_test


def z_normalize_fuc(data, test):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    z_normalized_data = (data - mean) / std
    z_normalized_test = (test - mean) / std
    return z_normalized_data, z_normalized_test


# Linear Support vector machines
class LSVM(AlgorithmBasic):
    def __init__(self, info, C, k ,center=False):
        super().__init__(info)
        if center:
            self.info.data, self.info.testData = center_data(self.info.data, self.info.testData)
        self.N = info.data.shape[1]
        self.K = k
        self.C = C
        self.DataLabelZ = np.array(list(map(lambda x: 1 if x == 1 else -1, self.info.label)))
        self.TestLabelZ = np.array(list(map(lambda x: 1 if x == 1 else -1, self.info.testlable)))
        self.spaceD = np.vstack((self.info.data, self.K * np.ones(self.N)))
        self.GMatrix = np.dot(self.spaceD.T, self.spaceD)
        self.LabelZMatrix = np.dot(self.DataLabelZ.reshape(-1, 1), self.DataLabelZ.reshape(1, -1))
        self.Hmatrix = self.GMatrix * self.LabelZMatrix
        bounds = [(0, self.C)] * self.N
        self.m, self.primal, _ = scipy.optimize.fmin_l_bfgs_b(func=self.LDc_obj,
                                                              bounds=bounds,
                                                              x0=np.zeros(self.N), factr=1.0)
        self.wc_star = np.sum(self.m * self.DataLabelZ * self.spaceD, axis=1)

    pass

    def LDc_obj(self, alpha):  # alpha has shape (n,)
        n = len(alpha)
        minusJDc = 0.5 * np.dot(np.dot(alpha.T, self.Hmatrix), alpha) - np.dot(alpha.T, np.ones(n))  # 1x1
        return minusJDc, self.gradLDc(alpha)

    def gradLDc(self, alpha):
        n = len(alpha)
        return (np.dot(self.Hmatrix, alpha) - 1).reshape(n)

    def applyTest(self):
        self.w = self.wc_star[:-1]
        self.b = self.wc_star[-1]
        self.S = np.dot(self.w.T, self.info.testData) + self.b * self.K
        self.primal_loss = self.primal_obj()
        self.dual_loss = -self.LDc_obj(self.m)[0]
        self.duality_gap = self.primal_obj() - self.dual_loss

        pass

    def primal_obj(self):
        return 0.5 * np.linalg.norm(self.wc_star) ** 2 + self.C * np.sum(
            np.maximum(0, 1 - self.DataLabelZ * np.dot(self.wc_star.T, self.spaceD)))

    def checkAcc(self):
        predict_labels = np.where(self.S > 0, 1, 0)
        return self.info.testlable == predict_labels
        pass


if __name__ == "__main__":
    DCFs = []
    min_DCFs = []
    listC = np.logspace(-5, 0, 11)
    listK = [1]
    for c in listC:
        for k in listK:
            KFold_ = KFold(5, prior=0.1, pca=0)
            LinearSVM = LSVM(KFold_.infoSet[0], c, k , center=True)
            LinearSVM.applyTest()
            # print('Primal loss: %f,Dual loss: %f, Duality gap: %.9f' % (
            #     LinearSVM.primal_loss, LinearSVM.dual_loss, LinearSVM.duality_gap))
            KFold_.addscoreList(LinearSVM.checkAcc())
            KFold_.addLLR(LinearSVM.S)
            # listScore = np.concatenate((listScore, LinearSVM.C))
            DCF, min_DCF = KFold_.ValidatClassfier('Centralized LinerSVM  C=%.5f, K=1' % (
                c), fold_number=0, threshold=0.5)

            DCFs.append(DCF)
            min_DCFs.append(min_DCF)

    plt.figure(figsize=(10, 7))
    plt.xscale("log", base=10)
    plt.plot(listC, DCFs, label=f'DCF')
    plt.plot(listC, min_DCFs, '--', label=f'Min DCF')
    plt.xlabel('C')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.title('DCF vs C')
    plt.legend()
    plt.grid(True)
    plt.show()
