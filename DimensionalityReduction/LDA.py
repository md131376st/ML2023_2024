import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from DimensionalityReduction.PCA import PCA


class LDA:
    def __init__(self, m, test_size=0.2, random_state=None, data=None, labels=None):
        self.m = m
        self.test_size = test_size
        self.random_state = random_state

        self.data = None
        self.labels = None
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None

        self.SampleSizeDiffClass = None
        self.eachClassMean = None
        self.sb = None
        self.sw = None
        if not data.any():
            self.LoadData()
        else:
            self.data = data
            self.labels = labels
        self.SplitData()
        self.NumberOfClassSample()
        self.SumAllDataValues = np.sum(self.SampleSizeDiffClass)
        self.SampleMean = self.CalculateMean(self.train_data)
        self.CalculateClassMean()
        self.CalculateSb()
        self.CalculateSW()
        self.CalculateEigenValueJoinSb()
        self.FixOrientation()
        # self.Threshold = self.CalculateThreshold()
        # self.PlotHistograms()
        self.Threshold = -0.1
        self.error_rate = self.ComputeErrorRate()
        print(f"Validation Threshold {self.Threshold} Error Rate: {self.error_rate}")

    def SplitData(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.arange(self.data.shape[1])
        np.random.shuffle(indices)

        split_point = int((1 - self.test_size) * len(indices))
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        self.train_data = self.data[:, train_indices]
        self.train_labels = self.labels[train_indices]
        self.val_data = self.data[:, val_indices]
        self.val_labels = self.labels[val_indices]

    def CalculateEigenValue(self):
        # Generalized eigenvalue problem
        s, U = scipy.linalg.eigh(self.sb, self.sw)
        W = U[:, ::-1][:, :self.m]
        # Find base value
        UW, _, _ = np.linalg.svd(W)
        self.eigenVector = UW[:, :self.m]

    def CalculateEigenValueJoinSb(self):
        # Generalized eigenvalue problem
        U, s, _ = np.linalg.svd(self.sw)
        p1 = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(s))), U.T)
        sbt = np.dot(np.dot(p1, self.sb), p1.T)
        s, P2 = np.linalg.eigh(sbt)
        P2 = P2[:, ::-1][:, :self.m]

        self.eigenVector = np.dot(p1.T, P2)

    def FixOrientation(self):
        projection_list = np.dot(self.eigenVector.T, self.train_data)
        mean_false = projection_list[:, self.train_labels == 0].mean()
        mean_true = projection_list[:, self.train_labels == 1].mean()
        if mean_false > mean_true:
            self.eigenVector = -self.eigenVector

    def CalculateThreshold(self):
        projection_list = np.dot(self.eigenVector.T, self.train_data)
        mean_false = projection_list[:, self.train_labels == 0].mean()
        mean_true = projection_list[:, self.train_labels == 1].mean()
        return (mean_false + mean_true) / 2

    def ComputeErrorRate(self):
        val_projection = np.dot(self.eigenVector.T, self.val_data)
        predictions = val_projection > self.Threshold
        error_rate = np.mean(predictions != self.val_labels)
        return error_rate

    def PlotHistograms(self):
        projection_list = np.dot(self.eigenVector.T, self.data)
        fake = projection_list[:, self.labels == 0].flatten()
        genuine = projection_list[:, self.labels == 1].flatten()

        plt.hist(fake, bins=30, alpha=0.5, label='fake', color='blue')
        plt.hist(genuine, bins=30, alpha=0.5, label='genuine', color='green')
        plt.axvline(x=self.Threshold, color='red', linestyle='--', label='Threshold')
        plt.xlabel('Projected LDA value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.title('Histogram of LDA projected samples')
        plt.show()

    def NumberOfClassSample(self):
        self.SampleSizeDiffClass = np.array([np.sum(self.train_labels == i) for i in np.unique(self.train_labels)])

    def CalculateClassMean(self):
        self.eachClassMean = np.array(
            [self.CalculateMean(self.train_data[:, self.train_labels == i]) for i in np.unique(self.train_labels)]).T

    def CalculateDiffMeanClassAndMeanDataset(self):
        return self.eachClassMean - self.VectorCol(self.SampleMean)

    def CalculateSb(self):
        self.sb = np.zeros((self.train_data.shape[0], self.train_data.shape[0]))
        for i in range(len(self.SampleSizeDiffClass)):
            diff_means = self.CalculateDiffMeanClassAndMeanDataset()[:, i:i + 1]
            self.sb += self.SampleSizeDiffClass[i] * np.dot(diff_means, diff_means.T)
        self.sb /= self.SumAllDataValues

    def CalculateSW(self):
        self.sw = np.zeros((self.train_data.shape[0], self.train_data.shape[0]))
        for i in range(len(self.SampleSizeDiffClass)):
            classData = self.train_data[:, self.train_labels == i]
            centerData = classData - self.VectorCol(self.eachClassMean.T[i])
            SWc = 1 / self.SampleSizeDiffClass[i] * np.dot(centerData, centerData.T)
            self.sw += self.SampleSizeDiffClass[i] * SWc
        self.sw /= self.SumAllDataValues

    def LoadData(self):
        data = np.genfromtxt("../Data/Train.txt", delimiter=",")
        self.labels = data[:, -1].astype(int)
        self.data = data[:, :-1].T

    def VectorCol(self, data):
        return data.reshape((data.size, 1))

    def VectorRow(self, data):
        return data.reshape((1, data.size))

    def CalculateMean(self, data):
        return data.mean(axis=1)


# lda = LDA(1, test_size=0.2, random_state=42)
# threshold_list = [0.1, -0.1, 0.5, -1 , -0.5]
# for thresh in threshold_list:
#     lda.Threshold = thresh
#     error_rate =lda.ComputeErrorRate()
#     print(f"Validation Threshold {lda.Threshold} Error Rate: {error_rate}")

# pca = PCA(5)
# lda = LDA(1, test_size=0.2, random_state=42, data=pca.projection_list, labels=pca.label)
# pca = PCA(4)
# lda = LDA(1, test_size=0.2, random_state=42, data=pca.projection_list, labels=pca.label)
# pca = PCA(3)
# lda = LDA(1, test_size=0.2, random_state=42, data=pca.projection_list, labels=pca.label)
# pca = PCA(2)
# lda = LDA(1, test_size=0.2, random_state=42, data=pca.projection_list, labels=pca.label)