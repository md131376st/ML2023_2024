import os

import numpy as np
from matplotlib import pylab
from DimensionalityReduction.Utility import VectorCol


class PCA:

    def __init__(self, m):
        # we want the lorgest m egin venctor
        self.m = m
        self.egin_vector = []
        self.mean = 0
        self.conversion = 0
        self.data = []
        self.label = []
        self.LoadData()
        self.CalculateMean()
        self.CenterData()
        self.ConversionMatrix()
        self.Eigenvectors()
        self.PlotFunction()

    def LoadData(self):
        self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)))
                                  + "/Train.txt",
                                  delimiter=",")
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
        sumAllEgineVectors= sum(sum(self.egin_vector))
        self.egin_vector = self.egin_vector[:, 0:self.m]
        sumSelectedEginVector= sum(sum (self.egin_vector))
        print("preserve Data Perservation wih " +str(self.m) +" : "+
              str((sumSelectedEginVector/sumAllEgineVectors)*100))
        self.projection_list = np.dot(self.egin_vector.T, self.data)

    def PlotFunction(self):
        # self.projection_list = np.dot(self.egin_vector.T, self.data)
        men = self.projection_list[:, self.label == 0]
        women = self.projection_list[:, self.label == 1]

        pylab.scatter(men[0, :], men[1, :])
        pylab.scatter(women[0, :], women[1, :])
        # pylab.scatter(projection_list[0],projection_list[1])
        pylab.xlabel('Principal component 1')
        pylab.ylabel('Principal component 2')
        pylab.legend(['men', 'women', "allData"])
        pylab.show()


pca = PCA(11)
pca1 = PCA(10)
pca2 = PCA(9)
pca3 = PCA(8)
pca4 = PCA(7)
pca5 = PCA(6)
pca6 = PCA(5)
pca7 = PCA(4)
pca8 = PCA(3)
pca9 = PCA(2)
