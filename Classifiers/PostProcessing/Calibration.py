import numpy as np
from matplotlib import pyplot as plt

from Classifiers.GaussionMixtureModels.GMMEM import GMMEM
from Classifiers.LogicalRegrationModels.BLR import BLR
from Classifiers.SuperVectorMachineBinary.KSVM import KSVM
from Data.Info import KFold


def vcol(x):
    return x.reshape((x.size, 1))


def genrate_calibration_dataset(classifiers, labels, kfold):
    score_list = []
    for classifier, label in zip(classifiers, labels):
        classifier.applyTest()
        scores = vcol(np.array(classifier.llr))
        labels = vcol(kfold.infoSet[0].testlable)
        data = np.hstack((scores, labels))
        filename = f"{label}_calibration_data.npy"
        np.save(filename, data)
        score_list.append(data)
    return score_list


def genrate_score_level_dataset(classifiers, kfold):
    score_list = []
    for classifier in classifiers:
        classifier.applyTest()
        score_list.append(classifier.llr)
    scores = np.column_stack(score_list)
    labels = vcol(kfold.infoSet[0].testlable)
    data = np.hstack((scores, labels))
    filename = f"score_level_fusion1.npy"
    np.save(filename, data)

    return data


def load_calibration_data(label):
    filename = f"{label}.npy"
    data = np.load(filename)
    return data


def Bayes_error_plot(labels, prior_colibration):
    # cal_dataset = genrate_calibration_dataset(classifiers, labels, kfold_)

    plt.figure(figsize=(10, 7))
    for label, prior in zip(labels, prior_colibration):
        data = load_calibration_data(label)
        prior_odds_range = np.linspace(-4, 4, 31)
        DCFs = []
        min_DCFs = []
        for prior_odds in prior_odds_range:
            pi = 1 / (1 + np.exp(-prior_odds))
            kfold_ = KFold(5, prior=pi, cfn=1, cfp=1, pca=0, data=data)
            for i in range(kfold_.k):
                logRegObj = BLR(kfold_.infoSet[i], 0, pi_T=prior)
                logRegObj.applyTest()
                kfold_.addscoreList(logRegObj.checkAcc())
                kfold_.addLLR(logRegObj.llr)
            DCF, min_DCF = kfold_.ValidatClassfier(f"{label}")
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
    # kfold_ = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0)
    # gmm_d = GMMEM(kfold_.infoSet[0], thresholdForEValues=0.01, numberOfComponents=16, model="diagonal")
    # gmm_tied = GMMEM(kfold_.infoSet[0], thresholdForEValues=0.01, numberOfComponents=32, model="tied")
    # kfold_1 = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0)
    # quadratic = BLR(kfold_1.infoSet[0], 0.0031622776601683794, quadratic=True)
    # kfold_2 = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0)
    # quadratic_1 = BLR(kfold_2.infoSet[0], 0.0001, quadratic=True)
    # kernel_SVM = KSVM(kfold_.infoSet[0], 'RBF', 1, 31.622776601683793, 1, np.exp(-2))
    # kernel_SVM1 = KSVM(kfold_.infoSet[0], 'RBF', 1, 100, 1, np.exp(-2))
    # classifiers = [gmm_d,
    #                gmm_tied,
    #                quadratic,
    #                quadratic_1,
    #                kernel_SVM,
    #                kernel_SVM1
    #                ]
    # labels = ['GMM Diagonal with 16 ',
    #           'GMM Tied with 32',
    #           'Quadratic Regression with lambda =0.0031622776601683794 ',
    #           'Quadratic Regression with lambda =0.0001',
    #           'Kernel SVM=RBF  with k=1, y=e^(-2), C=31.622776601683793',
    #           'kernel SVM=RVF with k=1  y=e^(-2) C=100'
    #           ]
    # top3_classifiers = [
    #     gmm_d,
    #     gmm_tied,
    #     kernel_SVM
    # ]
    # cal_dataset = genrate_score_level_dataset(classifiers, kfold_)
    # prior_colibration = [0.08839967720705841,
    #                      0.15886910488091516,
    #                      0.08839967720705841,
    #                      0.15886910488091516,
    #                      0.15886910488091516,
    #                      0.15886910488091516
    #                      ]
    # Bayes_error_plot(labels, prior_colibration)
    labels = ["score_level_fusion_top_6",
              "score_level_fusion_top_3"
              ]
    for label in labels:
        data = load_calibration_data(label)
        prior_odds_range = np.linspace(-3, 3, 11)
        # DCFs = []
        # min_DCFs = []
        for prior_odds in prior_odds_range:
            pi = 1 / (1 + np.exp(-prior_odds))
            kfold_ = KFold(5, prior=0.1, cfn=1, cfp=1, pca=0, data=data)
            for i in range(kfold_.k):
                logRegObj = BLR(kfold_.infoSet[i], 0, pi_T=pi)
                logRegObj.applyTest()
                kfold_.addscoreList(logRegObj.checkAcc())
                kfold_.addLLR(logRegObj.llr)
            DCF, min_DCF = kfold_.ValidatClassfier(f"{label} with prior:{pi}")
