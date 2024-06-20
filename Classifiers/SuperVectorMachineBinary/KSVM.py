import numpy as np

from Classifiers.algorithemsBasic import AlgorithmBasic
from Data.Info import KFold
import scipy.optimize
import scipy.special


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


# Kernel SVM
class KSVM(AlgorithmBasic):
    def __init__(self, info, kernelType, K, C, *params):

        super().__init__(info)
        self.params = params
        self.eps = params[0]
        self.kernel = 0
        self.K = K
        self.C = C
        self.N = self.info.data.shape[1]
        self.kernelType = kernelType
        if self.kernelType == 'Polynomial':
            self.polynomial(self.params, self.info.data, self.info.data)
        elif self.kernelType == 'RBF':
            self.RBF(self.params, self.info.data, self.info.data)
        self.DataLabelZ = np.array(list(map(lambda x: 1 if x == 1 else -1, self.info.label)))
        self.TestLabelZ = np.array(list(map(lambda x: 1 if x == 1 else -1, self.info.testlable)))
        self.spaceD = np.vstack([self.info.data, np.ones((1, self.N)) * self.K])
        self.LabelZMatrix = np.dot(self.DataLabelZ.reshape(-1, 1), self.DataLabelZ.reshape(1, -1))
        self.Hmatrix = self.LabelZMatrix * self.kernel
        bounds = [(0, self.C)] * self.N
        self.m, _, _ = scipy.optimize.fmin_l_bfgs_b(func=self.LDc_obj,
                                                    bounds=bounds,
                                                    x0=np.zeros(self.N),
                                                    factr=1.0)
        self.wc_star = np.sum(self.m * self.DataLabelZ * self.spaceD, axis=1)
        print('SVM (kernel) - C %e - dual loss %e' % (self.C, -self.LDc_obj(self.m)[0]))

        # return kernel + eps

    def polynomial(self, params, data, data1):
        self.c = params[1]
        self.d = params[2]
        self.kernel = (np.dot(data.T, data1) + self.c) ** self.d

    def RBF(self, params, data, data1):
        self.gamma = params[1]
        self.x = vcol((data ** 2).sum(0))
        self.y = vrow((data1 ** 2).sum(0))
        self.kernel = np.exp(
            -self.gamma * (self.x + self.y - 2 * np.dot(data.T, data1)))

    def applyTest(self):
        self.w = self.wc_star[:-1]
        self.b = self.wc_star[-1]
        if self.kernelType == 'Polynomial':
            self.polynomial(self.params, self.info.data, self.info.testData)
        elif self.kernelType == 'RBF':
            self.RBF(self.params, self.info.data, self.info.testData)
        self.S = np.dot((self.m * self.DataLabelZ).reshape(1, -1), self.kernel).T

        pass

    def checkAcc(self):
        predict_labels = np.where(self.S > 0, 1, 0).ravel()
        hi = self.info.testlable == predict_labels
        return self.info.testlable == predict_labels
        pass

    def LDc_obj(self, alpha):  # alpha has shape (n,)
        n = len(alpha)
        Ha = self.Hmatrix @ vcol(alpha)
        hi = np.dot(vrow(alpha), Ha)

        minusJDc = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()  # 1x1
        return minusJDc, self.gradLDc(alpha)

    def gradLDc(self, alpha):
        n = len(alpha)
        hi = (np.dot(self.Hmatrix, alpha) - 1).reshape(n)
        return (np.dot(self.Hmatrix, alpha) - 1).reshape(n)


if __name__ == "__main__":
    # listeps = [0, 1]
    # listGama = [1, 10]
    # listC = [1]
    # for eps in listeps:
    #     for c in listC:
    #         for gamma in listGama:
    #             KFold_ = KFold(3, prior=0.5, pca=0)
    #             LinearSVM = KSVM(KFold_.infoSet[2], 'RBF', eps, c, eps, gamma)
    #             LinearSVM.applyTest()
    #             KFold_.addscoreList(LinearSVM.checkAcc())
    #             KFold_.addLLR(LinearSVM.S)
    #             KFold_.ValidatClassfier('KernelSVM RBF  C=%.1f, K=%f, eps=%f, gamma=%f ' % (
    #                 c, eps, eps, gamma), fold_number=2, threshold=0)
    # np.savetxt("kernelRunRBF"+ str(c)+"_eps"+str(eps)+"_gamma"+str(gamma)+".txt",listScore )
    listeps = [0, 1]
    d = [2]
    clist = [0, 1]
    listC = [1]
    for eps in listeps:
        for c in listC:
            for d_ in d:
                for c_ in clist:
                    KFold_ = KFold(3, prior=0.5, pca=0)
                    LinearSVM = KSVM(KFold_.infoSet[2], 'Polynomial', eps, c, eps, c_, d_)
                    LinearSVM.applyTest()
                    KFold_.addscoreList(LinearSVM.checkAcc())
                    KFold_.addLLR(LinearSVM.S)
                    KFold_.ValidatClassfier('KernelSVM Polynomial  C=%.1f, K=%f, eps=%f, c=%f ,d=%f' % (
                        c, eps, eps, c_, d_), fold_number=2, threshold=0)
