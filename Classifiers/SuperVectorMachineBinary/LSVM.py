import numpy as np
from algorithemsBasic import AlgorithmBasic
from Info import KFold
import scipy.optimize
import scipy.special



# Linear Support vector machines
class LSVM(AlgorithmBasic):
    def __init__(self, info, C, k):
        super().__init__(info)
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
        self.dual_loss = self.LDc_obj(self.m)[0]
        self.duality_gap = self.primal_obj() + self.dual_loss

        pass

    def primal_obj(self):
        return 0.5 * np.linalg.norm(self.wc_star) ** 2 + self.C * np.sum(
            np.maximum(0, 1 - self.DataLabelZ * np.dot(self.wc_star.T, self.spaceD)))

    def duality_gap(self, alpha_star):
        return self.primal_obj() + self.LDc_obj(alpha_star)[0]

    def checkAcc(self):
        predict_labels = np.where(self.S > 0, 1, 0)
        return self.info.testlable == predict_labels
        pass


if __name__ == "__main__":
    KFold = KFold(3)
    listC = [0.1, 1, 10]
    listK = [1, 10]
    for c in listC:
        for k in listK:
            listScore=[]
            for i in range(KFold.k):
                LinearSVM = LSVM(KFold.infoSet[i], c, k)
                LinearSVM.applyTest()
                print('Primal loss: %f,Dual loss: %f, Duality gap: %.9f' % (
                    LinearSVM.primal_loss, LinearSVM.dual_loss, LinearSVM.duality_gap))
                KFold.addscoreList(LinearSVM.checkAcc())
                listScore=np.concatenate(( listScore,LinearSVM.C))
            KFold.ValidatClassfier('LinerSVM C=%.1f, K=%d' % (
                c, k))
            np.savetxt("LinerSVM"+ str(c)+"_k"+str(k)+".txt",listScore )
            KFold.scoreList = []
            # calibration
            for i in range(KFold.k):
                pass



