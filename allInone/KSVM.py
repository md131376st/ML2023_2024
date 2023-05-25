import numpy as np
from algorithemsBasic import AlgorithmBasic
from Info import KFold
import scipy.optimize
import scipy.special


# Kernel SVM
class KSVM(AlgorithmBasic):
    def __init__(self, info, kernelType, K, C, *params):

        super().__init__(info)
        self.eps = params[0]
        self.kernel = 0
        self.K = K
        self.C = C
        self.N = self.info.data.shape[1]
        if kernelType == 'Polynomial':
            self.c = params[1]
            self.d = params[2]
            self.kernel = (np.dot(self.info.data.T, self.info.data) + self.c) ** self.d
        elif kernelType == 'RBF':
            gamma = params[1]
            self.x = np.repeat(self.info.data, self.info.data.shape[1], axis=1)
            self.y = np.tile(self.info.data, self.info.data.shape[1])
            self.kernel = np.exp(
                -gamma * np.linalg.norm(self.x - self.y, axis=0).reshape(self.info.data.shape[1],
                                                                         self.info.data.shape[1]) ** 2)
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

        # return kernel + eps

    def applyTest(self):
        self.w = self.wc_star[:-1]
        self.b = self.wc_star[-1]
        self.S = np.sum(np.dot((self.m * self.DataLabelZ).reshape(1, -1), self.kernel + self.eps))

        pass

    def checkAcc(self):
        predict_labels = np.where(self.S > 0, 1, 0)
        return self.info.testlable == predict_labels
        pass

    def LDc_obj(self, alpha):  # alpha has shape (n,)
        n = len(alpha)
        minusJDc = 0.5 * np.dot(np.dot(alpha.T, self.Hmatrix), alpha) - np.dot(alpha.T, np.ones(n))  # 1x1
        return minusJDc, self.gradLDc(alpha)

    def gradLDc(self, alpha):
        n = len(alpha)
        return (np.dot(self.Hmatrix, alpha) - 1).reshape(n)


if __name__ == "__main__":
    KFold = KFold(3)
    listeps = [0, 1]
    listGama = [10 ** -3, 0.1, 1]
    listC = [0.1, 1, 2]
    for eps in listeps:
        for c in listC:
            for gamma in listGama:
                listScore=[]
                for i in range(KFold.k):
                    LinearSVM = KSVM(KFold.infoSet[i],  'RBF',c, 1, eps, gamma)
                    LinearSVM.applyTest()
                    KFold.addscoreList(LinearSVM.checkAcc())
                    listScore=np.concatenate(( listScore,LinearSVM.C))
                    # KFold.addRealScore(LinearSVM.S)
                KFold.ValidatClassfier('KernelSVM RBF  C=%.1f, K=1, eps=%f, gamma=%f ' % (
                    c, eps, gamma))
                np.savetxt("kernelRunRBF"+ str(c)+"_eps"+str(eps)+"_gamma"+str(gamma)+".txt",listScore )
                KFold.scoreList = []
                # calibration
                for i in range(KFold.k):
                    pass
