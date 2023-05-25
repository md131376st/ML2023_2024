import numpy
import scipy.special

from algorithemsBasic import AlgorithmBasic
from Info import KFold


class GMMEM(AlgorithmBasic):
    def __init__(self, info=None, thresholdForEValues=0, numberOfComponents=1, model="full"):
        super().__init__(info)
        self.thresholdForEValues = thresholdForEValues
        self.numberOfComponents = numberOfComponents
        self.model = model
        self.GMM_EM_wrapper()

    def imperical_mean(self):
        self.mu = self.make_col_matrix(self.info.data.mean(1))

    def make_col_matrix(self, row):
        return row.reshape(row.size, 1)

    def Cov_matrix(self, matrix):
        return numpy.dot(matrix, matrix.T) / matrix.shape[1]

    def normalize_data(self):
        return self.info.data - self.mu

    def GMM_EM_wrapper(self):
        self.imperical_mean()
        self.C = self.Cov_matrix(self.normalize_data())
        gmm_init_0 = [(1.0, self.mu, self.C)]
        NewgmmEM = 0
        itteration = 1 + int(numpy.log2(self.numberOfComponents))

        self.mu_Cov_weight_pair_each_class = {}

        for label in set(list(self.info.label)):
            # print("label=", label)
            gmm_init = gmm_init_0
            for i in range(itteration):
                # print("GMM LEN", len(gmm_init))
                NewgmmEM = self.GMM_EM(self.info.data[:, self.info.label == label], gmm_init)
                if i < itteration - 1:
                    gmm_init = self.gmmlbg(NewgmmEM, 0.1)
            self.mu_Cov_weight_pair_each_class[label] = NewgmmEM

    def applyTest(self):
        final = numpy.zeros((len(set(self.info.testlable)), self.info.testData.shape[1]))
        for i in set(list(self.info.label)):
            GMM = self.mu_Cov_weight_pair_each_class[i]
            SM = self.GMM_ll_PerSample(GMM)
            final[int(i)] = SM

        density = numpy.exp(final)
        self.llr = numpy.log(density[1, :] / density[0, :])
        self.predictedLabelByGMM = final.argmax(0)
        self.error = (self.predictedLabelByGMM == self.info.testlable).sum() / self.info.testlable.size
        return

    def checkAcc(self):
        return self.info.testlable == self.predictedLabelByGMM
        pass

    def Log_pdf_MVG_ND(self, X, mu, C):
        Y = [self.logpdf_ONEsample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]

        return numpy.array(Y).ravel()

    def make_row_matrix(self, row):
        return row.reshape(1, row.size)

    def logpdf_ONEsample(self, x, mu, C):
        P = numpy.linalg.inv(C)
        res = -0.5 * x.shape[0] * numpy.log(2 * numpy.pi)
        res += -0.5 * numpy.linalg.slogdet(C)[1]
        # error
        res += -0.5 * numpy.dot(numpy.dot((x - mu).T, P), x - mu)
        return res.ravel()

    def GMM_EM(self, X, GMM):
        llNew = None
        llOld = None
        G = len(GMM)
        N = X.shape[1]
        gmmnew = None
        while llOld is None or (llNew - llOld) > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = self.Log_pdf_MVG_ND(X, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            llNew = SM.sum() / N
            gmmnew = GMM
            P = numpy.exp(SJ - SM)
            gmmNew = []
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (self.make_row_matrix(gamma) * X).sum(1)
                S = numpy.dot(X, (self.make_row_matrix(gamma) * X).T)
                w = Z / N
                mu = self.make_col_matrix(F / Z)
                Sigma = S / Z - numpy.dot(mu, mu.T)
                # diag sigma
                if self.model == "diagonal":
                    Sigma = Sigma * numpy.eye(Sigma.shape[0])
                # to apply threshold for EValues
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < self.thresholdForEValues] = self.thresholdForEValues
                SigmaNew = numpy.dot(U, self.make_col_matrix(s) * U.T)
                gmmNew.append((w, mu, SigmaNew))
            GMM = gmmNew
            # print(llNew)
        # print(llNew - llOld)
        return gmmnew

    def GMM_SJoint(self, GMM):
        G = len(GMM)
        N = self.info.testData.shape[1]
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = self.Log_pdf_MVG_ND(self.info.testData, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
        return SJ

    def GMM_ll_PerSample(self, GMM):
        G = len(GMM)
        N = self.info.testData.shape[1]
        S = numpy.zeros((G, N))
        for g in range(G):
            S[g, :] = self.Log_pdf_MVG_ND(self.info.testData, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
        return scipy.special.logsumexp(S, axis=0)

    def gmmlbg(self, GMM, alpha):
        G = len(GMM)
        newGMM = []
        for g in range(G):
            (w, mu, CovarianMatrix) = GMM[g]
            U, s, _ = numpy.linalg.svd(CovarianMatrix)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            newGMM.append((w / 2, mu - d, CovarianMatrix))
            newGMM.append((w / 2, mu + d, CovarianMatrix))
        return newGMM


if __name__ == "__main__":
    KFold = KFold(3)
    for i in range(KFold.k):
        GMMEM_ = GMMEM(KFold.infoSet[i], thresholdForEValues=0.5, numberOfComponents=2)
        GMMEM_.applyTest()
        KFold.addscoreList(GMMEM_.checkAcc())
    KFold.ValidatClassfier("GMMEM")
