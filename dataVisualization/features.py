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
features_fake = D[:, L == 0]
features_genuine = D[:, L == 1]
nameFeatures = ['Feature-1', 'Feature-2', 'Feature-3', 'Feature-4', 'Feature-5', 'Feature-6']
plt.figure()
fig, axs = plt.subplots(2, 3, layout="constrained")
row = 0
col = 0
for f in range(6):
    axs[row, col].hist(features_fake[f], bins=20, density=True, alpha=0.4)
    axs[row, col].hist(features_genuine[f], bins=20, density=True, alpha=0.4)
    XPlot = np.linspace(-5, 5, 2000).reshape(1, -1)  # N=1000 samples of M=1 features -> it is a row vector
    mu = np.mean(features_fake[f].reshape(-1, 1)).reshape(-1, 1)
    C = 1 / len(features_fake[f]) * np.dot(features_fake[f] - mu, (features_fake[f] - mu).T)
    print(f"{nameFeatures[f]} fake mean: {mu}  variance:{C}")
    y = logpdf_GAU_ND(XPlot, mu, C)
    axs[row, col].plot(XPlot.ravel(), np.exp(y))

    mu = np.mean(features_genuine[f].reshape(-1, 1)).reshape(-1, 1)
    C = 1 / len(features_genuine[f]) * np.dot(features_genuine[f] - mu, (features_genuine[f] - mu).T)
    print(f"{nameFeatures[f]} genuine mean: {mu}  variance:{C}")
    y = logpdf_GAU_ND(XPlot, mu, C)
    axs[row, col].plot(XPlot.ravel(), np.exp(y))
    axs[row, col].set_title(nameFeatures[f])
    row += 1
    if row == 2:
        col += 1
        row = 0

# fig.legend(['MGD fake', ',MGD genuine', 'fake', 'genuine'], loc="lower left")
#
# plt.savefig("fetures.png")
#
for f in range(6):
    plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(2, 3)
    row = 0
    col = 0
    for x in range(6):
        if f == x:
            row += 1
            if row == 2:
                col += 1
                row = 0
            continue
        plt.xlabel(nameFeatures[f])
        plt.ylabel(nameFeatures[x])
        axs[row, col].scatter(features_fake[f], features_fake[x], s=1, label='Fake')
        axs[row, col].scatter(features_genuine[f], features_genuine[x], s=1, label='Genuine')
        row += 1
        if row == 2:
            col += 1
            row = 0

    fig.legend(['fake', 'genuine'])
    plt.savefig(f"fetures_{f+1}.png")

