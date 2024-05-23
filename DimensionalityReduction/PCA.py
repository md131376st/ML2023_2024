import os

import numpy as np
import matplotlib.pyplot as plt
from DimensionalityReduction.Utility import VectorCol


class PCA:

    def __init__(self, m, data=None):
        # we want the lorgest m egin venctor
        self.m = m
        self.egin_vector = []
        self.mean = 0
        self.conversion = 0
        if data!=None:
            self.label = data[:, -1].T
            self.data = data[:, :-1].T
        else:
            self.data = []
            self.label = []
            self.LoadData()
        self.CalculateMean()
        self.CenterData()
        self.ConversionMatrix()
        self.Eigenvectors()
        # self.PlotFunction()

    def LoadData(self):
        self.data = np.genfromtxt( "./Train.txt",delimiter=",")
        print(self.data.shape)
        # self.data = np.random.rand(self.data.shape[0]).argsort()
        self.label = self.data[:, -1].T
        self.data = self.data[:, :-1].T

    def CalculateMean(self):
        self.mean = self.data.mean(1)

    def CenterData(self):
        self.data = self.data - VectorCol(self.mean)

    def ConversionMatrix(self):
        self.conversion = np.dot(self.data, self.data.T) / self.data.shape[1]

    def Eigenvectors(self):
        self.egin_vector, s, vh = np.linalg.svd(self.conversion)
        sumAllEgineVectors = sum(sum(abs(self.egin_vector)))
        self.egin_vector = self.egin_vector[:, 0:self.m]
        sumSelectedEginVector = sum(sum(abs(self.egin_vector)))
        print("preserve Data Perservation wih " + str(self.m) + " : " +
              str((sumSelectedEginVector / sumAllEgineVectors) * 100))
        self.projection_list = np.dot(self.egin_vector.T, self.data)

    def PlotFunction(self):
        fake = self.projection_list[:, self.label == 0]
        genuine = self.projection_list[:, self.label == 1]
        for i in range(6):
            plt.figure(figsize=(8, 6))
            plt.hist(fake[i,:], bins=30, alpha=0.5, label=f'fake')
            plt.hist(genuine[i,:], bins=30, alpha=0.5, label=f'genuine')
            plt.title(f'Histogram of PCA Component {i + 1}')
            plt.xlabel('Projected Feature Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig("pca_" + str(i+1) + ".png")


pca = PCA(m=6)