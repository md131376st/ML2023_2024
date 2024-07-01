import os
import numpy as np
import sklearn.datasets

from DimensionalityReduction.PCA import PCA
import pandas as pd
from functools import reduce
import scipy.stats
import copy


def vrow(x):
    return x.reshape((1, x.size))


def vcol(x):
    return x.reshape((x.size, 1))


class Info:
    def __init__(self, data=None, test=None, isKfold=False):
        if not isKfold:
            self.LoadData()
        else:
            self.data = copy.deepcopy(data)
            self.test = copy.deepcopy(test)
        self.label = self.data[:, -1].T
        self.data = self.data[:, :-1].T

        self.testData = self.test[:, :-1].T
        self.testlable = self.test[:, -1].T

        self.Accoracy = 0
        self.err = 0

    def LoadData(self):
        self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__))) + "/Train.txt",
                                  delimiter=",")
        self.test = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__))) + "/evalData.txt",
                                  delimiter=",")

    def TransferData(self):
        pass

    def CheckAccuracy(self, correct_assign):
        self.Accoracy = correct_assign / len(self.testlable)

    def CalculateErrorRate(self):
        self.err = 1 - self.Accoracy



class KFold:
    def __init__(self, k, prior=0.5, cfn=1, cfp=1, pca=0, slice=None, start=None, data=None):
        self.k = k
        self.foldList = []
        self.pca = pca
        self.data = data
        self.LoadData(slice, start)
        self.infoSet = []
        self.lables = []
        self.GenerateInfoDataWithTest()
        self.minth = None
        self.scoreList = []
        self.realScore = []
        self.allFoldLabels = []
        nClasses = len(set(self.lables))
        self.ConfusionMatrices = np.zeros((nClasses, nClasses), dtype=np.int32)
        self.FNR = 0
        self.FPR = 0
        self.DCF = 0
        self.normalDCF = 0
        self.pi = prior
        self.cfn = cfn
        self.cfp = cfp
        self.LLR = np.array([])
        # self.effective_prior = ((self.pi * self.cfn)/( (self.pi * self.cfn)+((1-self.pi) * self.cfp) ))
        # print(f"Effective Prior: {self.effective_prior} ")
        pass

    def LoadData(self, slice, start):
        # self.test_iris()
        if self.data is None:
            self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)))
                                      + "/Train.txt",
                                      delimiter=",")
        if slice != None and start == None:
            self.data = np.concatenate((self.data[:, :slice], self.data[:, -1:]), axis=1)
        elif slice != None and start != None:
            end = start + slice
            self.data = np.concatenate((self.data[:, start:end], self.data[:, -1:]), axis=1)
        # Data has been shuffled before splitting, so
        # that the data of different folds are homogeneous
        dataPandas = pd.DataFrame(self.data)
        dataPandas = dataPandas.sample(
            frac=1,
            random_state=100,
        ).reset_index(drop=True)
        self.data = dataPandas.to_numpy()

        if self.pca != 0:
            self.data = np.hstack(
                (PCA(self.pca, self.data).projection_list.T, self.data[:, -1].reshape(1, self.data.shape[0]).T))
        self.foldsize = int(self.data.shape[0] / self.k)
        for i in range(self.k):
            if i == self.k - 1:
                self.foldList.append(self.data[i * self.foldsize:, :])
            else:
                self.foldList.append(self.data[i * self.foldsize:self.foldsize * (i + 1), :])
            pass

    def test_iris(self):
        D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
        D = D[:, L != 0]  # We remove setosa from D
        L = L[L != 0]  # We remove setosa from L
        L[L == 2] = 0
        self.data = np.concatenate((D, vrow(L)), axis=0)

        np.random.seed(0)
        idx = np.random.permutation(self.data.shape[1])
        self.data = self.data[:, idx].T

    def GenerateInfoDataWithTest(self):
        for i in range(self.k):
            test = self.foldList[i]
            self.lables = np.concatenate((self.lables, test[:, -1].T))
            data = np.zeros(shape=(0, self.data.shape[1]))
            for j in range(i):
                data = np.concatenate((data, self.foldList[j]))
            for j in range(i + 1, self.k):
                data = np.concatenate((data, self.foldList[j]))
            self.infoSet.append(Info(data, test, True))

    def addscoreList(self, scores):
        self.scoreList = np.concatenate((self.scoreList, scores))

    def addLLR(self, LLR):
        self.LLR = np.append(self.LLR, LLR)

    def addRealScore(self, scores):
        self.realScore = np.concatenate((self.realScore, scores))

    def binaryOptimalBayesDecision(self, threshold=None):
        self.LLR.flatten()
        if threshold == None:
            threshold = -1 * np.log((self.pi * self.cfn) / ((1 - self.pi) * self.cfp))
        score = [1 if self.LLR[i] >
                      threshold else 0 for i in range(self.LLR.size)]
        return np.array(score)

    def CheckAccuracy(self):
        self.Accoracy = sum(self.scoreList) / len(self.lables)

    def ValidatClassfier(self, classfierName, threshold=None, fold_number=None):
        if fold_number != None:
            self.lables = self.infoSet[fold_number].testlable
        self.CheckAccuracy()
        self.CalculateErrorRate()
        if threshold:
            self.realScore = self.binaryOptimalBayesDecision(None)
            self.scoreList = self.checkscore(self.realScore)

        self.binaryBayesRisk()
        dcf = self.DCF
        normal_dcf = self.normalDCF
        minDFC = self.binaryMinDCF()
        print(classfierName + ':  Error rate %f%%  ' % (
                self.err * 100) + ' Actual DCF ' + str(normal_dcf) + ' Min DCF ' + str(
            minDFC))

        return normal_dcf, minDFC

        # print("Min normalize DCF " + str(minDFC))

    def checkscore(self, new_score):
        return self.lables == new_score

    def binaryBayesRisk(self):
        self.CalculateConfusionMatrices()
        self.CalculateFNR()
        self.CalculateFPR()
        self.CalculateDCF()
        self.compute_normalized_DCF()

    def compute_Pfn_Pfp_allThresholds_fast(self):
        llrSorter = np.argsort(self.LLR)
        llrSorted = self.LLR[llrSorter]
        classLabelsSorted = self.lables[llrSorter]
        Pfp = []
        Pfn = []
        nTrue = (classLabelsSorted == 1).sum()
        nFalse = (classLabelsSorted == 0).sum()
        nFalseNegative = 0
        nFalsePositive = nFalse

        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

        for idx in range(len(llrSorted)):
            if classLabelsSorted[idx] == 1:
                nFalseNegative += 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
            if classLabelsSorted[idx] == 0:
                nFalsePositive -= 1  # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
            Pfn.append(nFalseNegative / nTrue)
            Pfp.append(nFalsePositive / nFalse)

        llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

        PfnOut = []
        PfpOut = []
        thresholdsOut = []
        for idx in range(len(llrSorted)):
            if idx == len(llrSorted) - 1 or llrSorted[idx + 1] != llrSorted[
                idx]:
                PfnOut.append(Pfn[idx])
                PfpOut.append(Pfp[idx])
                thresholdsOut.append(llrSorted[idx])

        return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)

    def binaryMinDCF(self):
        Pfn, Pfp, th = self.compute_Pfn_Pfp_allThresholds_fast()
        minDCF = (self.pi * self.cfn * Pfn + (1 - self.pi) * self.cfp * Pfp) / np.minimum(self.pi * self.cfn, (
                1 - self.pi) * self.cfp)
        idx = np.argmin(minDCF)
        # thresholds = np.concatenate([np.array([-np.inf]), self.LLR, np.array([np.inf])])
        # min_DCF = None
        # for i in thresholds:
        #     self.realScore = self.binaryOptimalBayesDecision(i)
        #     self.scoreList = self.checkscore(self.binaryOptimalBayesDecision(i))
        #     self.binaryBayesRisk()
        #     if min_DCF is None or self.normalDCF < min_DCF:
        #         min_DCF = self.normalDCF
        #         self.minth = i
        # print(normalizeDCFs)

        return minDCF[idx]

    def CalculateErrorRate(self):
        self.err = 1 - self.Accoracy

    def CalculateConfusionMatrices(self):

        i = 0
        # print(len(self.scoreList))
        for correctPredication in self.scoreList:
            actual_label = int(self.lables[i])
            correctPredication = int(correctPredication)  # class 0,1
            if correctPredication:
                self.ConfusionMatrices[actual_label][actual_label] += 1
            elif actual_label:
                self.ConfusionMatrices[0][actual_label] += 1
            else:
                self.ConfusionMatrices[1][actual_label] += 1
            i += 1
            # print(i)

    def CalculateFNR(self):
        self.FNR = self.ConfusionMatrices[0, 1] / (self.ConfusionMatrices[0, 1] + self.ConfusionMatrices[1, 1])
        pass

    def CalculateFPR(self):
        self.FPR = self.ConfusionMatrices[1, 0] / (self.ConfusionMatrices[0, 0] + self.ConfusionMatrices[1, 0])
        pass

    def compute_normalized_DCF(self):
        self.normalDCF = self.DCF / np.minimum(self.pi * self.cfn, (1 - self.pi) * self.cfp)

    def CalculateDCF(self):
        self.DCF = self.pi * self.cfn * self.FNR + (1 - self.pi) * self.cfp * self.FPR
        pass
