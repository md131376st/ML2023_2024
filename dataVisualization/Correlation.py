import numpy
import seaborn as sns
import matplotlib.pyplot as plt


def LoadData():
    data = numpy.genfromtxt('./Train.txt', delimiter=",")
    return (data[:, -1].T, data[:, :-1].T)


def vcol(x):
    return x.reshape(x.size, 1)


def emprialmean(D):
    return vcol(D.mean(1))


def Covariance(D):
    mu = D.mean(1)
    DC = D - mu.reshape((mu.size, 1))
    C = numpy.dot(DC, DC.T) / DC.shape[1]
    return C


def normalize_data(matrix, mu):
    return matrix - mu


def computeCorrelationMatrix(fullFeatureMatrix):
    C = Covariance(normalize_data(fullFeatureMatrix, emprialmean(fullFeatureMatrix)))
    correlations = numpy.zeros((C.shape[1], C.shape[1]))
    for x in range(C.shape[1]):
        for y in range(C.shape[1]):
            correlations[x, y] = numpy.abs(C[x, y] / (numpy.sqrt(C[x, x]) * numpy.sqrt(C[y, y])))
    return correlations


def generateHeatmaps(datasets, filename, color, titles):
    num_datasets = len(datasets)
    plt.style.use("seaborn")

    # Create a subplot grid of appropriate size
    fig, axes = plt.subplots(nrows=1, ncols=num_datasets, figsize=(5 * num_datasets, 5))

    # Check if there is only one dataset (axes array will not be created in this case)
    if num_datasets == 1:
        axes = [axes]

    # Loop through datasets and create heatmaps
    for i, data in enumerate(datasets):
        sns.heatmap(data, linewidth=1, cmap=color[i], annot=True, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xticklabels([j + 1 for j in range(data.shape[1])])
        axes[i].set_yticklabels([j + 1 for j in range(data.shape[0])])

    plt.tight_layout()  # Adjust subplot params
    plt.savefig(filename)


L, D = LoadData()
features_Men = D[:, L == 0]
features_women = D[:, L == 1]
corrolation = computeCorrelationMatrix(D)
corrolationMen = computeCorrelationMatrix(features_Men)
corrolationWomen = computeCorrelationMatrix(features_women)
generateHeatmaps([corrolation, corrolationMen, corrolationWomen],
                 "correlation.png", ["YlGnBu", "Blues", "Reds"],
                 ["DataSet", "Men", "Women"])
