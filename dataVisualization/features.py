import numpy as np
import matplotlib.pyplot as plt


def LoadData():
    data = np.genfromtxt('./Train.txt', delimiter=",")
    return (data[:, -1].T, data[:, :-1].T)


def logpdf_GAU_ND_1sample(x, mu, C):
    M = x.shape[0]  # num of features of sample x
    mu = mu.reshape(M, 1)  # mean of the sample
    xc = x - mu  # x centered
    invC = np.linalg.inv(C)
    _, log_abs_detC = np.linalg.slogdet(C)
    return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]  # num of features
    N = x.shape[1]  # num of samples
    y = np.zeros(N)  # array of N scalar elements
    for i in range(N):
        density_xi = logpdf_GAU_ND_1sample(x[:, i:i + 1], mu, C)
        y[i] = density_xi
    return y


L, D = LoadData()
features_Men = D[:, L == 0]
features_women = D[:, L == 1]
nameFeatures = ['Feature-1', 'Feature-2', 'Feature-3', 'Feature-4', 'Feature-5', 'Feature-6', 'Feature-7', 'Feature-8',
                'Feature-9', 'Feature-10', 'Feature-11', 'Feature-12']
plt.figure()
fig, axs = plt.subplots(4, 3, layout="constrained")
row = 0
col = 0
for f in range(12):
    axs[row, col].hist(features_Men[f], bins=20, density=True, alpha=0.4)
    axs[row, col].hist(features_women[f], bins=20, density=True, alpha=0.4)
    XPlot = np.linspace(-35, 35, 2000).reshape(1, -1)  # N=1000 samples of M=1 features -> it is a row vector
    mu = np.mean(features_Men[f].reshape(-1, 1)).reshape(-1, 1)
    C = 1 / len(features_Men[f]) * np.dot(features_Men[f] - mu, (features_Men[f] - mu).T)
    y = logpdf_GAU_ND(XPlot, mu, C)
    axs[row, col].plot(XPlot.ravel(), np.exp(y))

    mu = np.mean(features_women[f].reshape(-1, 1)).reshape(-1, 1)
    C = 1 / len(features_women[f]) * np.dot(features_women[f] - mu, (features_women[f] - mu).T)
    y = logpdf_GAU_ND(XPlot, mu, C)
    axs[row, col].plot(XPlot.ravel(), np.exp(y))
    axs[row, col].set_title(nameFeatures[f])
    row +=1
    if row==4:
        col+=1
        row=0


fig.legend(['MGD men', ',MGD women', 'men', 'women'],  loc='outside upper right'
                 )

plt.savefig("fetures.png")
