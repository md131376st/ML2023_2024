import numpy as np
import matplotlib.pyplot as plt
from Data.Info import Info
from DimensionalityReduction.PCA import PCA


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]  # num of features
    N = x.shape[1]  # num of samples
    y = []
    # y = np.zeros(N)  # array of N scalar elements
    invC = np.linalg.inv(C)
    _, log_abs_detC = np.linalg.slogdet(C)
    const = (M * np.log(2 * np.pi)) + log_abs_detC
    for i in range(N):
        newX = x[:, i:i + 1]
        mu = mu.reshape(M, 1)  # mean of the sample
        xc = newX - mu  # x centered
        density_xi = (const + np.dot(np.dot(xc.T, invC), xc)) * (-0.5)
        y.append(density_xi)
    # hi = np.array(y).reshape(M,N)
    return np.array(y).ravel()


def Calculate_GAU(data):
    mean = data.mean(1)
    covarianceMatrix = np.cov(data)
    y = logpdf_GAU_ND(data, mean, covarianceMatrix)
    plt.figure()
    XPlot = np.linspace(-8, 12, data.shape[1])
    plt.plot(XPlot.ravel(), np.exp(y))
    plt.show()


def MaxLiklihoodEstimate(data):
    # compute mu_ML
    # we add one to have at least 1 attribute for each feature
    mu_ML = data.mean(1).reshape(-1, 1) + 1
    # compute sigma_ML
    centerData = data - mu_ML
    sigma_ML = 1 / data.shape[1] * np.dot(centerData, centerData.T)
    return mu_ML, sigma_ML, logpdf_GAU_ND(data, mu_ML, sigma_ML)


# apply MVG
# Calculate_GAU(data.projection_list)
# apply Maximum Likelihood Estimate
def checkMVGwithPCA():
    # first reduce the feature dimentiality
    data = PCA(4)
    mu_ML, sigma_ML, y = MaxLiklihoodEstimate(data.projection_list)
    plt.figure()
    # plt.hist(data.projection_list.ravel(), bins=50, density=True)
    plt.hist(data.projection_list[:, data.label == 0].ravel(), bins=100, density=True)
    plt.hist(data.projection_list[:, data.label == 1].ravel(), bins=100, density=True)
    XPlot = np.linspace(-8, 12, data.projection_list.shape[1]).reshape(1, -1)
    # compute the density
    np.exp(y)
    plt.plot(XPlot.ravel(), np.exp(y))
    plt.show()


def MVG():
    data = Info().data
    mu_ML, sigma_ML, y = MaxLiklihoodEstimate(data)
    plt.figure()
    # plt.hist(data.projection_list.ravel(), bins=50, density=True)
    plt.hist(data.ravel(), bins=100, density=True)
    XPlot = np.linspace(-8, 12, data.shape[1]).reshape(1, -1)

    plt.plot(XPlot.ravel(), np.exp(y))
    plt.show()

MVG()