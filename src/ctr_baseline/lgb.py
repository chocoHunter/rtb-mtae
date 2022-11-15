import os
import sys
sys.path.append('../')
import time
import logging
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from config import base_config
from util.processor import Processor, EncodeData

logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)


class LgbCtrPredictor:
    """
    LgbCtrPredictor class is a custom lightgbm CTR predictor.
    """
    def __init__(self,):
        """
        Initialize the LgbCtrPredictor class.
        """
        self.label = 'click'
        self.model = lgb.LGBMClassifier(objective='binary',
                                   num_leaves=64,
                                   learning_rate=0.01,
                                   n_estimators=2000,
                                   colsample_bytree=0.65,
                                   subsample=0.75,
                                   n_jobs=-1,
                                   seed=2018
                                   # reg_alpha = 0.4
                                   )

    def fit(self, X_train, Y_train, X_test, Y_test):
        """
        Fit the lightgbm ctr predictor
        :param X_train: train label encoded feature: Dataframe
        :param Y_train: train click label: list, np.array
        :param X_test: test label encoded feature: Dataframe
        :param Y_test: test click label: list, np.array
        :return: lightgbm model
        """
        logger.info("Start fitting lgb_ctr_predictor")
        fit_start_time = time.time()
        self.model.fit(
            X_train,
            Y_train,
            eval_set=[
                (X_test,
                 Y_test)],
            eval_metric='binary_logloss',
            early_stopping_rounds=100)
        fit_end_time = time.time()
        logger.info("Fitting lgb_ctr_predictor done. Time used: {}s".format(fit_end_time-fit_start_time))

        train_auc, train_logloss = self.evalute(X_train, Y_train)
        test_auc, test_logloss = self.evalute(X_test, Y_test)
        logger.info('Train classify auc is: {}%'.format(train_auc * 100))
        logger.info('Test classify auc is: {}%'.format(test_auc * 100))
        logger.info('Train logloss is: {}%'.format(train_logloss))
        logger.info('Test logloss is: {}%'.format(test_logloss))
        report = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_logloss': train_logloss,
            'test_logloss': test_logloss
        }
        return report

    def evalute(self, X, Y):
        """
        Evalute the auc metric
        :param X: label encoded feature: Dataframe
        :param Y: click label: list, np.array
        :return:
        """
        pred = self.model.predict_proba(X, )[:, 1]
        auc = metrics.roc_auc_score(Y, pred)
        logloss = metrics.log_loss(Y, pred)
        return auc, logloss


if __name__ == '__main__':
    dataset = 'ipinyou'
    # campaign_list = base_config.campaign_list[dataset]
    campaign_list = ['1458']
    report_df = pd.DataFrame()
    for camp in campaign_list:
        # load data
        processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        train_encode = processor.load_encode('train')
        test_encode = processor.load_encode('test')
        X_train, Y_train = train_encode.X, train_encode.Y
        X_test, Y_test = test_encode.X, test_encode.Y

        lgb_ctr_predictor = LgbCtrPredictor()
        report = lgb_ctr_predictor.fit(X_train, Y_train, X_test, Y_test)
        report['camp'] = camp
        report_df = report_df.append([report], ignore_index=True)

    report_dir = base_config.log_path + 'ctr_baseline/ctr_lgb/'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    report_df.to_excel(report_dir + '{}_report.xlsx'.format(dataset))