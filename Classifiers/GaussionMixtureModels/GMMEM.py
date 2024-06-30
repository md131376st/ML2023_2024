import numpy
import scipy.special
from matplotlib import pyplot as plt

from Classifiers.algorithemsBasic import AlgorithmBasic
from Data.Info import KFold, Info


class GMMEM(AlgorithmBasic):
    def __init__(self, info=None, thresholdForEValues=0.0, numberOfComponents=1, model="full"):
        """model can be on of this values: full,diagonal,tied """
        super().__init__(info)
        self.all_info = self.info
        self.info_list = []
        self.thresholdForEValues = thresholdForEValues
        self.numberOfComponents = numberOfComponents
        self.model = model
        self.GMM_EM_wrapper()

    def imperical_mean(self, data):
        return self.make_col_matrix(data.mean(1))

    def make_col_matrix(self, row):
        return row.reshape(row.size, 1)

    def Cov_matrix(self, matrix):
        return numpy.dot(matrix, matrix.T) / matrix.shape[1]

    def normalize_data(self, data, mu):
        return data - mu

    def GMM_EM_wrapper(self):

        NewgmmEM = 0
        self.mu_Cov_weight_pair_each_class = {}
        for label in set(list(self.info.label)):
            class_data = self.info.data[:, self.info.label == label]
            mu = self.imperical_mean(class_data)
            C = self.Cov_matrix(self.normalize_data(class_data, mu))
            if self.model == "diagonal":
                C = C * numpy.eye(class_data.shape[0])
            C = self.constraining_eigin_value(C)
            gmm_init_0 = [(1.0, mu, C)]
            gmm_init = gmm_init_0
            while len(gmm_init) < self.numberOfComponents:
                gmm_init = self.gmmlbg(gmm_init, 0.1)
                gmm_init = self.GMM_EM(self.info.data[:, self.info.label == label], gmm_init)
            self.mu_Cov_weight_pair_each_class[label] = gmm_init

    def applyTest(self):
        final = numpy.zeros((len(set(self.info.testlable)), self.info.testData.shape[1]))
        for i in set(list(self.info.label)):
            GMM = self.mu_Cov_weight_pair_each_class[i]
            _, SM = self.logpdf_GMM(self.info.testData, GMM)
            final[int(i)] = SM

        self.llr = final[1] - final[0]
        self.predictedLabelByGMM = final.argmax(0)
        self.error = (self.predictedLabelByGMM == self.info.testlable).sum() / self.info.testlable.size
        return

    def checkAcc(self):
        return self.info.testlable == self.predictedLabelByGMM
        pass

    def Log_pdf_MVG_ND(self, X, mu, C):
        # Y = [self.logpdf_ONEsample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]
        # return numpy.array(Y).ravel()
        # """faster version """
        P = numpy.linalg.inv(C)
        return -0.5 * X.shape[0] * numpy.log(numpy.pi * 2) - 0.5 * numpy.linalg.slogdet(C)[1] - 0.5 * (
                (X - mu) * (P @ (X - mu))).sum(0)

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
        new_gmm = GMM
        while llOld is None or (llNew - llOld) > 1e-6:
            llOld = llNew
            SJ, SM = self.logpdf_GMM(X, new_gmm)
            llNew = SM.sum() / N
            # posterior probability
            P = numpy.exp(SJ - SM)
            # M-step
            update_gmm = []
            for g in range(G):
                # update for each component
                gamma = P[g, :]
                # zero order
                Z = gamma.sum()
                # First order
                F = (self.make_row_matrix(gamma) * X).sum(1)
                # second order
                S = numpy.dot(X, (self.make_row_matrix(gamma) * X).T)
                w = Z / N
                mu = self.make_col_matrix(F / Z)
                Sigma = S / Z - numpy.dot(mu, mu.T)
                # diag sigma
                if self.model == "diagonal":
                    Sigma = Sigma * numpy.eye(Sigma.shape[0])
                if self.model != "tied":
                    Sigma = self.constraining_eigin_value(Sigma)
                update_gmm.append((w, mu, Sigma))
            if self.model == "tied":
                CTied = 0
                for w, mu, C in update_gmm:
                    CTied += w * C
                Sigma = self.constraining_eigin_value(CTied)
                update_gmm = [(w, mu, Sigma) for w, mu, C in update_gmm]

            new_gmm = update_gmm

        return new_gmm

    def constraining_eigin_value(self, Sigma):
        if self.thresholdForEValues > 0:
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < self.thresholdForEValues] = self.thresholdForEValues
            Sigma = numpy.dot(U, self.make_col_matrix(s) * U.T)
        return Sigma

    def logpdf_GMM(self, X, new_gmm):
        G = len(new_gmm)
        N = X.shape[1]
        SJ = numpy.zeros((G, N))
        # E-step
        for g in range(G):
            SJ[g, :] = self.Log_pdf_MVG_ND(X, new_gmm[g][1], new_gmm[g][2]) + numpy.log(new_gmm[g][0])
        # log-marginal fx(x)
        SM = scipy.special.logsumexp(SJ, axis=0)
        return SJ, SM

    def GMM_SJoint(self, GMM):
        G = len(GMM)
        N = self.info.testData.shape[1]
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = self.Log_pdf_MVG_ND(self.info.testData, GMM[g][1], GMM[g][2]) + numpy.log(GMM[g][0])
        return SJ

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
    componentlist = [ 16]
    DCFs = []
    min_DCFs = []
    for component in componentlist:
        KFold_ = KFold(5, prior=0.1, pca=0)
        # GMMEM_ = GMMEM(KFold_.infoSet[0], thresholdForEValues=0.01, numberOfComponents=component)
        GMMEM_ = GMMEM(KFold_.infoSet[0], thresholdForEValues=0.01, numberOfComponents=component, model="diagonal")
        GMMEM_.applyTest()
        KFold_.addscoreList(GMMEM_.checkAcc())
        KFold_.addLLR(GMMEM_.llr)
        DCF, min_DCF = KFold_.ValidatClassfier(f"GMM Tied  with component: {component}",
                                               fold_number=0, threshold=0)
        DCFs.append(DCF)
        min_DCFs.append(min_DCF)

    plt.figure(figsize=(10, 7))
    plt.plot(componentlist, DCFs, label=f'DCF')
    plt.plot(componentlist, min_DCFs, '--', label=f'Min DCF')
    plt.xticks(componentlist, componentlist)
    plt.xlabel('component')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.title('DCF vs C')
    plt.legend()
    plt.grid(True)
    plt.show()
