import numpy 
import seaborn
import matplotlib.pyplot as plot

def LoadData():
        data = numpy.genfromtxt('./Train.txt', delimiter=",")
        return (data[:, -1].T, data[:, :-1].T)

L, D = LoadData()

def vcol(x):
    return x.reshape(x.size, 1)

def emprialmean(D):
    return vcol(D.mean(1))


def Covariance(D):
    mu = D.mean(1)
    DC = D - mu.reshape((mu.size, 1))
    C = numpy.dot(DC, DC.T)/ DC.shape[1]
    return C

def normalize_data(matrix,mu):
    return matrix-mu
 
def computeCorrelationMatrix(fullFeatureMatrix):
    C = Covariance(normalize_data(fullFeatureMatrix, emprialmean(fullFeatureMatrix)))
    correlations = numpy.zeros((C.shape[1], C.shape[1]))
    for x in range(C.shape[1]):
        for y in range(C.shape[1]):
            correlations[x,y] = numpy.abs( C[x,y] / ( numpy.sqrt(C[x,x]) * numpy.sqrt(C[y,y]) ) )
    return correlations

def heatMap(data):
    plot.style.use("seaborn")
    plot.figure(figsize=(data.shape[1], data.shape[1]))
    heat_map = seaborn.heatmap(data, linewidth=1,cmap="YlGnBu", annot=True)
    plot.title("HeatMap for gender detection project features")
    plot.savefig("correlation.png")


corrolation =computeCorrelationMatrix(D)
heatMap(corrolation)