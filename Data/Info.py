import os
import numpy as np
from DimensionalityReduction.PCA import PCA
import scipy.stats


class Info:
    def __init__(self, data=None, test=None, isKfold=False):
        if not isKfold:
            self.LoadData()
        else:
            self.data = data
            self.test = test
        self.label = self.data[:, -1].T
        self.data = self.data[:, :-1].T

        self.testData = self.test[:, :-1].T
        self.testlable = self.test[:, -1].T

        self.Accoracy = 0
        self.err = 0

    def LoadData(self):
        self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__))) + "/Train.txt",
                                  delimiter=",")
        self.test = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__))) + "/Test.txt", delimiter=",")

    def TransferData(self):
        pass

    def CheckAccuracy(self, correct_assign):
        self.Accoracy = correct_assign / len(self.testlable)

    def CalculateErrorRate(self):
        self.err = 1 - self.Accoracy

    def ValidatClassfier(self, sum_correct_assign, classfierName):
        self.CheckAccuracy(sum_correct_assign)
        self.CalculateErrorRate()
        print(classfierName + ':  Error rate %f%%' % (self.err * 100))


class KFold:
    def __init__(self, k, prior=0.5, cfn=1, cfp=1):
        self.k = k
        self.foldList = []
        self.LoadData()
        self.infoSet = []
        self.lables = []
        self.GenerateInfoDataWithTest()

        self.scoreList = []
        self.realScore = []
        self.allFoldLabels = []
        self.ConfusionMatrices = np.zeros(shape=(len(set(self.lables)), len(set(self.lables))))
        self.FNR = 0
        self.FPR = 0
        self.DCF = 0
        self.normalDCF = 0
        self.pi = prior
        self.cfn = cfn
        self.cfp = cfp
        pass

    def LoadData(self):
        self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)))
                                  + "/Train.txt",
                                  delimiter=",")

        # self.data = np.hstack((PCA(6).projection_list.T, self.data[:, -1].reshape(1, self.data.shape[0]).T))
        self.size = int((self.data.shape[0]) / 2)
        self.wemonData = self.data[self.size:,:]
        self.MenData = self.data[:self.size,:]
        self.foldsize = int(self.size / self.k)
        for i in range(self.k):
            self.foldList.append(np.concatenate((self.MenData[i * self.foldsize:self.foldsize * (i + 1), :],
                                                 self.wemonData[i * self.foldsize:self.foldsize * (i + 1), :])))
            # lables = np.concatenate((self.MenData[i * self.foldsize:self.foldsize * (i + 1), -1],
            #                          self.wemonData[i * self.foldsize:self.foldsize * (i + 1), -1]))

            # self.allFoldLabels = np.concatenate((self.allFoldLabels, lables))
            pass

    def GenerateInfoDataWithTest(self):
        for i in range(self.k):

            test = self.foldList[i]
            self.lables = np.concatenate((self.lables, test[:, -1].T))
            data = np.zeros(shape=(0, self.data.shape[1]))
            for j in range(i):
                data = np.concatenate((data, self.foldList[i]))
            for j in range(i + 1, self.k):
                data = np.concatenate((data, self.foldList[j]))
            self.infoSet.append(Info(data, test, True))

    def addscoreList(self, scores):
        self.scoreList = np.concatenate((self.scoreList, scores))

    def addRealScore(self, scores):
        self.realScore = np.concatenate((self.realScore, scores))

    def CheckAccuracy(self):
        self.Accoracy = sum(self.scoreList) / len(self.lables)

    def ValidatClassfier(self, classfierName):
        self.CheckAccuracy()
        self.CalculateErrorRate()
        self.CalculateConfusionMatrices()
        self.CalculateFNR()
        self.CalculateFPR()
        self.CalculateDCF()
        self.compute_normalized_DCF()
        print(classfierName + ':  Error rate %f%%  ' % (
                self.err * 100) + 'DCF ' + str(self.DCF) + ' normal DCF ' + str(self.normalDCF))

    def CalculateErrorRate(self):
        self.err = 1 - self.Accoracy

    def CalculateConfusionMatrices(self):
        i = 0

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


hi = Info()
