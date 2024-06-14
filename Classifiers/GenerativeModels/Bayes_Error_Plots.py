import numpy as np
from matplotlib import pyplot as plt

from MGC import MGC
from NBG import NBG
from TCG import TCG
from Data.Info import KFold, Info


def expit(x):
    return 1 / (1 + np.exp(-x))


def plot_DCF(classifiers, prior_odds_range, labels):
    plt.figure(figsize=(10, 7))
    for classifier, label in zip(classifiers, labels):
        DCFs = []
        min_DCFs = []
        for prior_odds in prior_odds_range:
            pi = 1 / (1 + np.exp(-prior_odds))
            classifier.prior = pi
            kfold_ = KFold(k=k, prior=pi, cfn=1, cfp=1, pca=5)
            classifier.applyTest()
            kfold_.addscoreList(classifier.checkAcc())
            kfold_.addLLR(classifier.foldLLR)
            DCF, min_DCF = kfold_.ValidatClassfier(f"{classifier} prior:{pi}  cfn:1, cfp:1 pca:5", fold_number=0)
            DCFs.append(DCF)
            min_DCFs.append(min_DCF)

        plt.plot(prior_odds_range, DCFs, label=f'{label} DCF')
        plt.plot(prior_odds_range, min_DCFs, '--', label=f'{label} Min DCF')
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.legend()
    plt.grid(True)
    plt.title('DCF vs Prior Log-Odds')
    plt.show()


if __name__ == '__main__':
    k = 5
    kfold = KFold(k=k, prior=0.1, cfn=1, cfp=1, pca=5)

    # Assuming Info and KFold are already loaded with data
    info = Info(data=kfold.data, test=kfold.data, isKfold=True)

    mv_classifier = MGC(kfold.infoSet[0], 0.1)
    tied_classifier = TCG(kfold.infoSet[0], 0.1)
    naive_classifier = NBG(kfold.infoSet[0], 0.1)

    classifiers = [mv_classifier, tied_classifier, naive_classifier]
    labels = ['MVG', 'Tied MVG', 'Naive Bayes']

    prior_odds_range = np.linspace(-4, 4, 31)
    plot_DCF(classifiers, prior_odds_range, labels)

    pass
