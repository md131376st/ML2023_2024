# select model
import numpy as np
from matplotlib import pyplot as plt

from Classifiers import bayesRisk
from Classifiers.GaussionMixtureModels.GMMEM import GMMEM
from Classifiers.LogicalRegrationModels.BLR import BLR
from Classifiers.SuperVectorMachineBinary.KSVM import KSVM
from Data.Info import Info, KFold


def vcol(x):
    return x.reshape((x.size, 1))


def genrate_score_level_dataset_for_evaluation(classifiers, classifier_name,
                                               original_info, evaluation_info):
    score_list = []
    data = np.column_stack((original_info.data.T, original_info.label.T))
    test = np.column_stack((evaluation_info.testData.T, evaluation_info.testlable.T))
    for classifier, name in zip(classifiers, classifier_name):
        info = Info(data=data, test=test, isKfold=True)
        if name in ['Quadratic Regression with lambda =0.0031622776601683794 ',
                    'Quadratic Regression with lambda =0.0001']:
            info.testData = classifier.add_quadratic_features(info.testData)
        classifier.info = info
        classifier.applyTest()
        score_list.append(classifier.llr)

    scores = np.column_stack(score_list)
    classifier_name = vcol(evaluation_info.testlable)
    data = np.hstack((scores, classifier_name))
    filename = f"score_level_fusion_validation.npy"
    np.save(filename, data)

    return data


def genrate_score_level_dataset_training(classifiers, labels, label):
    score_list = []
    for classifier, name in zip(classifiers, labels):
        info = Info(isFiusion=True)
        if name in ['Quadratic Regression with lambda =0.0031622776601683794 ',
                    'Quadratic Regression with lambda =0.0001']:
            info.testData = classifier.add_quadratic_features(info.testData)

        classifier.info = info
        classifier.applyTest()
        score_list.append(classifier.llr)
    scores = np.column_stack(score_list)
    labels = vcol(label)
    data = np.hstack((scores, labels))
    filename = f"score_level_fusion_validation1.npy"
    np.save(filename, data)

    return data


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
    cal_dataset = genrate_score_level_dataset(classifiers, kfold_)
    info = Info(isKfold=False)
    evaluation_set = genrate_score_level_dataset_for_evaluation(classifiers, labels, kfold_.infoSet[0], info)
    prior = 0.08317269649392238
    info_cal_dataset = Info(data=cal_dataset, test=evaluation_set, isKfold=True)
    logRegObj = BLR(info_cal_dataset, 0, pi_T=prior)
    logRegObj.applyTest()
    print(
        'minDCF - pT = 0.1: %.4f' % bayesRisk.compute_minDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, 0.1,
                                                                         1.0, 1.0))
    print(
        'actDCF - pT = 0.1: %.4f' % bayesRisk.compute_actDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, 0.1,
                                                                         1.0, 1.0))
    # Bayes error plots
    plt.figure(figsize=(10, 7))
    DCFs = []
    min_DCFs = []
    prior_odds_range = np.linspace(-4, 4, 31)
    for prior_odds in prior_odds_range:
        pi = 1 / (1 + np.exp(-prior_odds))
        min_DCFs.append(bayesRisk.compute_minDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, pi, 1.0, 1.0))
        DCFs.append(bayesRisk.compute_actDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, pi, 1.0, 1.0))
    plt.plot(prior_odds_range, DCFs, label=f'score_level_fusion_top_6 DCF')
    plt.plot(prior_odds_range, min_DCFs, '--', label=f'score_level_fusion_top_6 Min DCF')
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.legend()
    plt.grid(True)
    plt.title('DCF vs Prior Log-Odds')
    plt.show()
    # analysing 3 top models
    top3_classifiers = [
        gmm_d,
        gmm_tied,
        kernel_SVM
    ]
    prior_colibration = [0.08839967720705841,
                         0.15886910488091516,
                         0.15886910488091516]
    top3_labels = [
        'GMM Diagonal with 16 ',
        'GMM Tied with 32',
        'Kernel SVM=RBF  with k=1, y=e^(-2), C=31.622776601683793',
    ]

    for model, prior, label in zip(top3_classifiers, prior_colibration, top3_labels):
        # generate data set for training calibration  the calibration
        model.info = kfold_.infoSet[0]
        model.applyTest()
        scores = vcol(model.llr)
        labels = vcol(kfold_.infoSet[0].testlable)
        data = np.hstack((scores, labels))
        info = Info(isKfold=False)
        # genrating evaluation set
        test = np.column_stack((info.testData.T, info.testlable.T))
        train_data = np.column_stack((kfold_.infoSet[0].data.T, kfold_.infoSet[0].label.T))
        # apply_test_first_level
        model.info = Info(data=train_data, test=test, isKfold=True)
        model.applyTest()
        scores = vcol(model.llr)
        test_data = np.hstack((scores, vcol(info.testlable)))
        info_cal_dataset = Info(data=data, test=test_data, isKfold=True)
        logRegObj = BLR(info_cal_dataset, 0, pi_T=prior)
        logRegObj.applyTest()
        print(
            f'{label} minDCF - pT = 0.1: %.4f' % bayesRisk.compute_minDCF_binary_fast(
                logRegObj.llr,
                info_cal_dataset.testlable,
                0.1,
                1.0, 1.0))
        print(
            f'{label} actDCF - pT = 0.1: %.4f' % bayesRisk.compute_actDCF_binary_fast(
                logRegObj.llr,
                info_cal_dataset.testlable,
                0.1,
                1.0, 1.0))
        plt.figure(figsize=(10, 7))
        DCFs = []
        min_DCFs = []
        prior_odds_range = np.linspace(-4, 4, 31)
        for prior_odds in prior_odds_range:
            pi = 1 / (1 + np.exp(-prior_odds))
            min_DCFs.append(
                bayesRisk.compute_minDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, pi, 1.0, 1.0))
            DCFs.append(bayesRisk.compute_actDCF_binary_fast(logRegObj.llr, info_cal_dataset.testlable, pi, 1.0, 1.0))
        plt.plot(prior_odds_range, DCFs, label=f'score_level_fusion_top_6 DCF')
        plt.plot(prior_odds_range, min_DCFs, '--', label=f'score_level_fusion_top_6 Min DCF')
        plt.xlabel('Prior Log-Odds')
        plt.ylabel('Detection Cost Function (DCF)')
        plt.legend()
        plt.grid(True)
        plt.title('DCF vs Prior Log-Odds')
        plt.show()
