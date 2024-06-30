import numpy as np
from matplotlib import pyplot as plt

from Classifiers.GaussionMixtureModels.GMMEM import GMMEM
from Classifiers.LogicalRegrationModels.BLR import BLR
from Classifiers.SuperVectorMachineBinary.KSVM import KSVM
from Data.Info import KFold


def expit(x):
    return 1 / (1 + np.exp(-x))


def plot_DCF(classifiers, prior_odds_range, labels):
    plt.figure(figsize=(10, 7))
    for classifier, label in zip(classifiers, labels):
        DCFs = []
        min_DCFs = []
        classifier.applyTest()
        for prior_odds in prior_odds_range:
            pi = 1 / (1 + np.exp(-prior_odds))
            kfold_ = KFold(k=5, prior=pi, cfn=1, cfp=1, pca=0)
            print(f"{label} with pi ={pi}")
            kfold_.addscoreList(classifier.checkAcc())
            kfold_.addLLR(classifier.llr)
            if labels == "GMM Diagonal with 16 " or label == "GMM Tied with 32":
                DCF, min_DCF = kfold_.ValidatClassfier(f"{classifier} prior:{pi}  cfn:1, cfp:1", fold_number=0)
            else:
                DCF, min_DCF = kfold_.ValidatClassfier(f"{classifier} prior:{pi}  cfn:1, cfp:1", fold_number=0,
                                                       threshold=0.5)
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
    kfold_ = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0)
    gmm_d = GMMEM(kfold_.infoSet[0], thresholdForEValues=0.01, numberOfComponents=16, model="diagonal")
    gmm_tied = GMMEM(kfold_.infoSet[0], thresholdForEValues=0.01, numberOfComponents=32, model="tied")
    kfold_1 = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0)
    quadratic = BLR(kfold_1.infoSet[0], 0.0031622776601683794, quadratic=True)
    kfold_2 = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0)
    quadratic_1 = BLR(kfold_2.infoSet[0], 0.0001, quadratic=True)
    kernel_SVM = KSVM(kfold_.infoSet[0], 'RBF', 1, 31.622776601683793, 1, np.exp(-2))
    kernel_SVM1 = KSVM(kfold_.infoSet[0], 'RBF', 1, 100, 1, np.exp(-2))

    classifiers = [gmm_d, gmm_tied
        , quadratic, quadratic_1, kernel_SVM, kernel_SVM1]
    labels = ['GMM Diagonal with 16 ',
              'GMM Tied with 32',

              'Quadratic Regression with lambda =0.0031622776601683794 ',
              'Quadratic Regression with lambda =0.0001',
              'Kernel SVM=RBF  with k=1, y=e^(-2), C=31.622776601683793',
              'kernel SVM=RVF with k=1  y=e^(-2) C=100'
              ]

    prior_odds_range = np.linspace(-4, 4, 31)
    plot_DCF(classifiers, prior_odds_range, labels)

    pass
