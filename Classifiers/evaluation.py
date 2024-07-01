# select model
import numpy as np
from matplotlib import pyplot as plt

from Classifiers import bayesRisk
from Classifiers.GaussionMixtureModels.GMMEM import GMMEM
from Classifiers.LogicalRegrationModels.BLR import BLR
from Classifiers.SuperVectorMachineBinary.KSVM import KSVM
from Data.Info import Info


def vcol(x):
    return x.reshape((x.size, 1))


def genrate_score_level_dataset(classifiers, label):
    score_list = []
    for classifier in classifiers:
        classifier.applyTest()
        score_list.append(classifier.llr)
    scores = np.column_stack(score_list)
    labels = vcol(label)
    data = np.hstack((scores, labels))
    filename = f"score_level_fusion_validation.npy"
    np.save(filename, data)

    return data


if __name__ == '__main__':
    info = Info()
    gmm_d = GMMEM(info, thresholdForEValues=0.01, numberOfComponents=16, model="diagonal")
    gmm_tied = GMMEM(info, thresholdForEValues=0.01, numberOfComponents=32, model="tied")
    info1 = Info()
    quadratic = BLR(info1, 0.0031622776601683794, quadratic=True)
    info2 = Info()
    quadratic_1 = BLR(info2, 0.0001, quadratic=True)
    kernel_SVM = KSVM(info, 'RBF', 1, 31.622776601683793, 1, np.exp(-2))
    kernel_SVM1 = KSVM(info, 'RBF', 1, 100, 1, np.exp(-2))
    classifiers = [gmm_d,
                   gmm_tied,
                   quadratic,
                   quadratic_1,
                   kernel_SVM,
                   kernel_SVM1
                   ]
    labels = ['GMM Diagonal with 16 ',
              'GMM Tied with 32',
              'Quadratic Regression with lambda =0.0031622776601683794 ',
              'Quadratic Regression with lambda =0.0001',
              'Kernel SVM=RBF  with k=1, y=e^(-2), C=31.622776601683793',
              'kernel SVM=RVF with k=1  y=e^(-2) C=100'
              ]
    cal_dataset = genrate_score_level_dataset(classifiers, info.label)
    prior = 0.08317269649392238
    info_cal_dataset = Info(data=cal_dataset, test=info.test, isKfold=True)
    logRegObj = BLR(info_cal_dataset, 0, pi_T=prior)
    logRegObj.applyTest()
    # used of provided code form labs

    print(
        'minDCF - pT = 0.1: %.4f' % bayesRisk.compute_minDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, 0.1,
                                                                         1.0, 1.0))
    print(
        'actDCF - pT = 0.1: %.4f' % bayesRisk.compute_actDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, 0.1,
                                                                         1.0, 1.0))
    # Bayes error plots
    # plt.figure(figsize=(10, 7))
    # DCFs = []
    # min_DCFs = []
    # prior_odds_range = np.linspace(-4, 4, 31)
    # for prior_odds in prior_odds_range:
    #     pi = 1 / (1 + np.exp(-prior_odds))
    #     min_DCFs.append(bayesRisk.compute_minDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, pi, 1.0, 1.0))
    #     DCFs.append(bayesRisk.compute_actDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, pi, 1.0, 1.0))
    # plt.plot(prior_odds_range, DCFs, label=f'score_level_fusion_top_6 DCF')
    # plt.plot(prior_odds_range, min_DCFs, '--', label=f'score_level_fusion_top_6 Min DCF')
    # plt.xlabel('Prior Log-Odds')
    # plt.ylabel('Detection Cost Function (DCF)')
    # plt.legend()
    # plt.grid(True)
    # plt.title('DCF vs Prior Log-Odds')
    # plt.show()
    pass
