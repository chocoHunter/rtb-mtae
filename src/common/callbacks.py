import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn import metrics
from scipy.stats import entropy, wasserstein_distance
from common.tool import count_pdf, count_cdf, avg_pdf, avg_cdf, plot_mp_result, plot_mp_result_regression, plot_mp_case, pdf2cdf
from common.metrics import anlp, anlp_fixed, anlp_z_prob
from common.batch_predict import batch_predict, mt_batch_predict, light_mp_batch_predict, light_mt_batch_predict


logger = logging.getLogger('tensorflow')


def evalute_ctr_metrics(Y_train, Y_test, Y_pred_train, Y_pred_test):
    train_loss = metrics.log_loss(Y_train, Y_pred_train)
    test_loss = metrics.log_loss(Y_test, Y_pred_test)
    train_auc = metrics.roc_auc_score(Y_train, Y_pred_train)
    test_auc = metrics.roc_auc_score(Y_test, Y_pred_test)
    ctr_metrics = {
        'train_log_loss': train_loss,
        'test_log_loss': test_loss,
        'train_auc': train_auc,
        'test_auc': test_auc
    }

    logger.info('Total Train Log Loss:{:.6f}. Total Log Test Loss:{:.6f}'.format(train_loss, test_loss))
    logger.info('Total Train  AUC:{:.6f}. Total Test  AUC:{:.6f}'.format(train_auc, test_auc))
    return ctr_metrics


def evalute_mp_metrics(Z_train, Z_test, Z_result_pred_train, Z_result_pred_test, predict_type='softmax', predict_z_size=301, save_fig=True, figure_path='', pdf_path='', epoch='', camp=''):
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)

    if predict_type == 'regression':
        Z_pred_train = Z_result_pred_train
        Z_pred_test = Z_result_pred_test
    elif predict_type == 'softmax':
        predict_pdf_train = Z_result_pred_train
        predict_pdf_test = Z_result_pred_test

    count_pdf_train = count_pdf(Z_train, predict_z_size)
    count_pdf_test = count_pdf(Z_test, predict_z_size)
    count_cdf_train = count_cdf(Z_train, predict_z_size)
    count_cdf_test = count_cdf(Z_test, predict_z_size)

    if predict_type == 'regression':
        count_pdf_pred_train = count_pdf(Z_pred_train, predict_z_size)
        count_pdf_pred_test = count_pdf(Z_pred_test, predict_z_size)
        count_cdf_pred_train = count_cdf(Z_pred_train, predict_z_size)
        count_cdf_pred_test = count_cdf(Z_pred_test, predict_z_size)
    elif predict_type == 'softmax':
        count_pdf_pred_train = avg_pdf(predict_pdf_train)
        count_pdf_pred_test = avg_pdf(predict_pdf_test)
        count_cdf_pred_train = avg_cdf(predict_pdf_train)
        count_cdf_pred_test = avg_cdf(predict_pdf_test)
        weights = np.array([i for i in range(predict_z_size)])
        Z_pred_train = predict_pdf_train.dot(weights)
        Z_pred_test = predict_pdf_test.dot(weights)

    train_mse = metrics.mean_squared_error(Z_train, Z_pred_train)
    test_mse = metrics.mean_squared_error(Z_test, Z_pred_test)
    train_mae = metrics.mean_absolute_error(Z_train, Z_pred_train)
    test_mae = metrics.mean_absolute_error(Z_test, Z_pred_test)
    train_kld = tf.keras.metrics.kullback_leibler_divergence(count_pdf_train, count_pdf_pred_train).numpy()
    test_kld = tf.keras.metrics.kullback_leibler_divergence(count_pdf_test, count_pdf_pred_test).numpy()
    train_re = entropy(count_pdf_train, count_pdf_pred_train)
    test_re = entropy(count_pdf_test, count_pdf_pred_test)
    train_wd = wasserstein_distance(count_pdf_train, count_pdf_pred_train)
    test_wd = wasserstein_distance(count_pdf_test, count_pdf_pred_test)
    if predict_type == 'regression':
        train_anlp = anlp_fixed(Z_train, count_pdf_pred_train)
        test_anlp = anlp_fixed(Z_test, count_pdf_pred_test)
    elif predict_type == 'softmax':
        train_anlp = anlp(Z_train, predict_pdf_train)
        test_anlp = anlp(Z_test, predict_pdf_test)

    mp_metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_kld': train_kld,
        'test_kld': test_kld,
        'train_re': train_re,
        'test_re': test_re,
        'train_wd': train_wd,
        'test_wd': test_wd,
        'train_anlp': train_anlp,
        'test_anlp': test_anlp
    }

    logger.info('Total Train ANLP:{:.6f}. Total Test ANLP:{:.6f}.'.format(train_anlp, test_anlp))
    logger.info('Total Train MSE:{:.6f}. Total TEST MSE:{:.6f}.'.format(train_mse, test_mse))
    logger.info('Total Train MAE:{:.6f}. Total Test MAE:{:.6f}.'.format(train_mae, test_mae))
    logger.info('Total Train KLD:{:.6f}. Total Test KLD:{:.6f}.'.format(train_kld, test_kld))
    logger.info('Total Train RE:{:.6f}. Total Test RE:{:.6f}.'.format(train_re, test_re))
    logger.info('Total Train WD:{:.6f}. Total Test WD:{:.6f}.'.format(train_wd, test_wd))

    if save_fig:
        if predict_type == 'softmax':
            train_pdf_title = 'Epoch {}: Camp {} train pdf'.format(epoch, camp)
            train_cdf_title = 'Epoch {}: Camp {} train cdf'.format(epoch, camp)
            test_pdf_title = 'Epoch {}: Camp {} test pdf'.format(epoch, camp)
            test_cdf_title = 'Epoch {}: Camp {} test cdf'.format(epoch, camp)
            case_study_pdf_title = 'Epoch {}: Camp {} test pdf case study'.format(epoch, camp)
            case_study_cdf_title = 'Epoch {}: Camp {} test cdf case study'.format(epoch, camp)
            plot_mp_result(count_pdf_train, count_pdf_pred_train, save_fig=True, figure_path=figure_path,
                           title=train_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
            plot_mp_result(count_cdf_train, count_cdf_pred_train, save_fig=True, figure_path=figure_path,
                           title=train_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)
            plot_mp_result(count_pdf_test, count_pdf_pred_test, save_fig=True, figure_path=figure_path,
                           title=test_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
            plot_mp_result(count_cdf_test, count_cdf_pred_test, save_fig=True, figure_path=figure_path,
                           title=test_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)
            case_index = np.random.randint(len(Z_test))
            case_z = Z_test[case_index]
            case_pdf = predict_pdf_test[case_index]
            case_cdf = pdf2cdf(case_pdf)
            pickle.dump(case_pdf, open(pdf_path + 'case_study_pdf_{}.pkl'.format(epoch), 'wb'))
            pickle.dump(case_cdf, open(pdf_path + 'case_study_cdf_{}.pkl'.format(epoch), 'wb'))
            plot_mp_case(case_z, case_pdf, save_fig=True, figure_path=figure_path,
                           title=case_study_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
            plot_mp_case(case_z, case_cdf, save_fig=True, figure_path=figure_path,
                         title=case_study_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        elif predict_type == 'regression':
            plot_mp_result_regression(Z_train, Z_pred_train, save_fig=True,
                                      figure_path=figure_path,
                                      title='Epoch {}: Camp {} train'.format(epoch, camp))
            plot_mp_result_regression(Z_test, Z_pred_test, save_fig=True,
                                      figure_path=figure_path,
                                      title='Epoch {}: Camp {} test'.format(epoch, camp))


    pickle.dump(count_pdf_pred_train, open(pdf_path + 'train_pdf_{}.pkl'.format(epoch), 'wb'))
    pickle.dump(count_pdf_pred_test, open(pdf_path + 'test_pdf_{}.pkl'.format(epoch), 'wb'))

    return mp_metrics


def light_evalute_mp_metrics(Z_train, Z_test, Z_pred_train, Z_pred_test, Z_prob_train, Z_prob_test, pdf_pred_train, pdf_pred_test, predict_z_size=301, save_fig=True, figure_path='', pdf_path='', epoch='', camp=''):
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
    count_pdf_train = count_pdf(Z_train, predict_z_size)
    count_pdf_test = count_pdf(Z_test, predict_z_size)
    count_cdf_train = count_cdf(Z_train, predict_z_size)
    count_cdf_test = count_cdf(Z_test, predict_z_size)
    cdf_pred_trian = pdf2cdf(pdf_pred_train)
    cdf_pred_test = pdf2cdf(pdf_pred_test)

    train_mse = metrics.mean_squared_error(Z_train, Z_pred_train)
    test_mse = metrics.mean_squared_error(Z_test, Z_pred_test)
    train_mae = metrics.mean_absolute_error(Z_train, Z_pred_train)
    test_mae = metrics.mean_absolute_error(Z_test, Z_pred_test)
    train_kld = tf.keras.metrics.kullback_leibler_divergence(count_pdf_train, pdf_pred_train).numpy()
    test_kld = tf.keras.metrics.kullback_leibler_divergence(count_pdf_test, pdf_pred_test).numpy()
    train_re = entropy(count_pdf_train, pdf_pred_train)
    test_re = entropy(count_pdf_test, pdf_pred_test)
    train_wd = wasserstein_distance(count_pdf_train, pdf_pred_train)
    test_wd = wasserstein_distance(count_pdf_test, pdf_pred_test)
    train_anlp = anlp_z_prob(Z_prob_train)
    test_anlp = anlp_z_prob(Z_prob_test)

    mp_metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_kld': train_kld,
        'test_kld': test_kld,
        'train_re': train_re,
        'test_re': test_re,
        'train_wd': train_wd,
        'test_wd': test_wd,
        'train_anlp': train_anlp,
        'test_anlp': test_anlp
    }

    logger.info('Total Train ANLP:{:.6f}. Total Test ANLP:{:.6f}.'.format(train_anlp, test_anlp))
    logger.info('Total Train MSE:{:.6f}. Total TEST MSE:{:.6f}.'.format(train_mse, test_mse))
    logger.info('Total Train MAE:{:.6f}. Total Test MAE:{:.6f}.'.format(train_mae, test_mae))
    logger.info('Total Train KLD:{:.6f}. Total Test KLD:{:.6f}.'.format(train_kld, test_kld))
    logger.info('Total Train RE:{:.6f}. Total Test RE:{:.6f}.'.format(train_re, test_re))
    logger.info('Total Train WD:{:.6f}. Total Test WD:{:.6f}.'.format(train_wd, test_wd))

    if save_fig:
        train_pdf_title = 'Epoch {}: Camp {} train pdf'.format(epoch, camp)
        train_cdf_title = 'Epoch {}: Camp {} train cdf'.format(epoch, camp)
        test_pdf_title = 'Epoch {}: Camp {} test pdf'.format(epoch, camp)
        test_cdf_title = 'Epoch {}: Camp {} test cdf'.format(epoch, camp)
        plot_mp_result(count_pdf_train, pdf_pred_train, save_fig=True, figure_path=figure_path,
                       title=train_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_result(count_cdf_train, cdf_pred_trian, save_fig=True, figure_path=figure_path,
                       title=train_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_result(count_pdf_test, pdf_pred_test, save_fig=True, figure_path=figure_path,
                       title=test_pdf_title, tf_summary=True, show_fig=False, epoch=epoch)
        plot_mp_result(count_cdf_test, cdf_pred_test, save_fig=True, figure_path=figure_path,
                       title=test_cdf_title, tf_summary=True, show_fig=False, epoch=epoch)


    pickle.dump(pdf_pred_train, open(pdf_path + 'train_pdf_{}.pkl'.format(epoch), 'wb'))
    pickle.dump(pdf_pred_test, open(pdf_path + 'test_pdf_{}.pkl'.format(epoch), 'wb'))

    return mp_metrics


def tf_summary_metrics(metrics, epoch):
    for key in metrics.keys():
        tf.summary.scalar(key, data=metrics[key], step=epoch)


class LogCTRLossCallback(tf.keras.callbacks.Callback):
    """
    LogCTRLossCallback class is used to log the information about the loss during the training of the CTR model.
    """
    def __init__(self, X_train, X_test, Y_train, Y_test, predict_batch_size, import_model=None, **kwargs):
        """
        Intialize LogCTRLossCallback class with full-volume data and batch size.
        :param X_train: train feature X: list
        :param X_test: test feature X: list
        :param Y_train: train label Y: list
        :param Y_test: test label Y: list
        :param predict_batch_size: batch size for predicting: int
        :param kwargs: kwargs of super class Callback
        """
        super(LogCTRLossCallback, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.predict_batch_size = predict_batch_size
        self.import_model = import_model
        self.result_df = pd.DataFrame()

    def on_epoch_end(self, epoch, logs):
        """
        Log the information about the metrics during the training of the CTR model at the end of each epoch.
        Logs include AUC and binary entropy loss.
        :param epoch: epoch: int
        :param logs: tensorflow default logs: dict
        :return: None
        """
        print('\n')
        logger.info('======================Current epoch:{}======================'.format(epoch))
        logger.info('=====================Evaluating metrics=====================')

        model = self.import_model if self.import_model else self.model
        Y_pred_train = batch_predict(model, self.X_train, self.predict_batch_size, output_dim=1)
        Y_pred_test = batch_predict(model, self.X_test, self.predict_batch_size, output_dim=1)

        ctr_metrics = evalute_ctr_metrics(self.Y_train, self.Y_test, Y_pred_train, Y_pred_test)
        tf_summary_metrics(ctr_metrics, epoch)
        ctr_metrics = dict(ctr_metrics, **logs)
        ctr_metrics['epoch'] = epoch
        self.result_df = self.result_df.append([ctr_metrics], ignore_index=True)


class LogMPLossCallback(tf.keras.callbacks.Callback):
    """
    LogMPLossCallback class is used to log the information about the loss during the training of the MP model.
    """
    def __init__(self, X_train, X_test, Z_train, Z_test,
                 predict_batch_size, predict_type='softmax', predict_z_size=301,
                 save_fig=False, figure_path='', pdf_path='', camp='', import_model=None, light_batch_predict=True, **kwargs):
        """
        Intialize LogMPLossCallback class with full-volume data and batch size.
        :param X_train: train feature X: list
        :param X_test: test feature X: list
        :param Z_train: train market price Z: list
        :param Z_test: test market price Z: list
        :param predict_batch_size: batch size for predicting: int
        :param kwargs: kwargs of super class Callback
        """
        super(LogMPLossCallback, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.Z_train = Z_train
        self.Z_test = Z_test
        self.predict_batch_size = predict_batch_size
        self.predict_z_size = predict_z_size
        self.predict_type = predict_type
        self.save_fig = save_fig
        self.figure_path = figure_path
        self.pdf_path = pdf_path
        self.camp = camp
        self.import_model = import_model
        self.light_batch_predict = light_batch_predict
        self.result_df = pd.DataFrame()

    def on_epoch_end(self, epoch, logs):
        """
        Log the information about the metrics during the training of the CTR model at the end of each epoch.
        Logs include MSE, MAE, ANLP, KLD, RE, WD.
        :param epoch: epoch: int
        :param logs: tensorflow default logs: dict
        :return: None
        """
        print('\n')
        logger.info('======================Current epoch:{}======================'.format(epoch))
        logger.info('=====================Evaluating metrics=====================')
        model = self.import_model if self.import_model else self.model

        if self.light_batch_predict:
            Z_pred_train, Z_prob_train, pdf_pred_train = light_mp_batch_predict(model, self.X_train, self.Z_train, self.predict_batch_size, output_dim=self.predict_z_size)
            Z_pred_test, Z_prob_test, pdf_pred_test = light_mp_batch_predict(model, self.X_test, self.Z_test,
                                                                                self.predict_batch_size,
                                                                                output_dim=self.predict_z_size)
            mp_metrics = light_evalute_mp_metrics(Z_train=self.Z_train, Z_test=self.Z_test, Z_pred_train=Z_pred_train, Z_pred_test=Z_pred_test, Z_prob_train=Z_prob_train, Z_prob_test=Z_prob_test, pdf_pred_train=pdf_pred_train, pdf_pred_test=pdf_pred_test, predict_z_size=self.predict_z_size,
                                            save_fig=self.save_fig, figure_path=self.figure_path, pdf_path=self.pdf_path, epoch=epoch, camp=self.camp)
        else:
            Z_result_pred_train = batch_predict(model, self.X_train, self.predict_batch_size, output_dim=self.predict_z_size)
            Z_result_pred_test = batch_predict(model, self.X_test, self.predict_batch_size, output_dim=self.predict_z_size)
            mp_metrics = evalute_mp_metrics(Z_train=self.Z_train, Z_test=self.Z_test, Z_result_pred_train=Z_result_pred_train, Z_result_pred_test=Z_result_pred_test,
                                            predict_type=self.predict_type, predict_z_size=self.predict_z_size,
                                            save_fig=self.save_fig, figure_path=self.figure_path, pdf_path=self.pdf_path, epoch=epoch, camp=self.camp)
        tf_summary_metrics(mp_metrics, epoch)
        mp_metrics['epoch'] = epoch
        mp_metrics = dict(mp_metrics, **logs)
        self.result_df = self.result_df.append([mp_metrics], ignore_index=True)


class LogMTLossCallback(tf.keras.callbacks.Callback):
    """
    LogMTLossCallback class is used to log the information about the loss during the training of the multi-task model.
    """
    def __init__(self, X_train, X_test, Y_train, Y_test, Z_train, Z_test,
                 predict_batch_size, predict_type='softmax', predict_z_size=301,
                 save_fig=False, figure_path='', pdf_path='', camp='', import_model=None, light_batch_predict=True, **kwargs):
        """
        Intialize LogMTLossCallback class with full-volume data and batch size.
        :param X_train: train feature X: list
        :param X_test: test feature X: list
        :param Y_train: train label Y: list
        :param Y_test: test label Y: list
        :param Z_train: train market price Z: list
        :param Z_test: test market price Z: list
        :param predict_batch_size: batch size for predicting: int
        :param kwargs: kwargs of super class Callback
        """
        super(LogMTLossCallback, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Z_train = Z_train
        self.Z_test = Z_test
        self.predict_batch_size = predict_batch_size
        self.predict_z_size = predict_z_size
        self.predict_type = predict_type
        self.save_fig = save_fig
        self.figure_path = figure_path
        self.pdf_path = pdf_path
        self.camp = camp
        self.import_model = import_model
        self.light_batch_predict = light_batch_predict
        self.result_df = pd.DataFrame()

    def on_epoch_end(self, epoch, logs):
        """
        Log the information about the loss during the training of the multi-task model at the end of each epoch.
        Logs include AUC, LOG-LOSS, MSE, MAE, ANLP, KLD, RE, WD.
        :param epoch: epoch: int
        :param logs: tensorflow default logs: dict
        :return: None
        """
        print('\n')
        logger.info('======================Current epoch:{}======================'.format(epoch))
        logger.info('=====================Evaluating metrics=====================')
        model = self.import_model if self.import_model else self.model
        if self.light_batch_predict:
            Y_pred_train, Z_pred_train, Z_prob_train, pdf_pred_train = light_mt_batch_predict(model, self.X_train, self.Z_train,
                                                                                self.predict_batch_size,
                                                                                output_dim=self.predict_z_size)
            Y_pred_test, Z_pred_test, Z_prob_test, pdf_pred_test = light_mt_batch_predict(model, self.X_test, self.Z_test,
                                                                             self.predict_batch_size,
                                                                             output_dim=self.predict_z_size)
            mp_metrics = light_evalute_mp_metrics(Z_train=self.Z_train, Z_test=self.Z_test, Z_pred_train=Z_pred_train,
                                                  Z_pred_test=Z_pred_test, Z_prob_train=Z_prob_train,
                                                  Z_prob_test=Z_prob_test, pdf_pred_train=pdf_pred_train,
                                                  pdf_pred_test=pdf_pred_test, predict_z_size=self.predict_z_size,
                                                  save_fig=self.save_fig, figure_path=self.figure_path,
                                                  pdf_path=self.pdf_path, epoch=epoch, camp=self.camp)
        else:
            Y_pred_train, Z_result_pred_train = mt_batch_predict(model, self.X_train, self.predict_batch_size, output_dim=self.predict_z_size)
            Y_pred_test, Z_result_pred_test = mt_batch_predict(model, self.X_test, self.predict_batch_size, output_dim=self.predict_z_size)
            mp_metrics = evalute_mp_metrics(Z_train=self.Z_train, Z_test=self.Z_test, Z_result_pred_train=Z_result_pred_train, Z_result_pred_test=Z_result_pred_test,
                                            predict_type=self.predict_type, predict_z_size=self.predict_z_size,
                                            save_fig=self.save_fig, figure_path=self.figure_path, pdf_path=self.pdf_path, epoch=epoch, camp=self.camp)
        ctr_metrics = evalute_ctr_metrics(self.Y_train, self.Y_test, Y_pred_train, Y_pred_test)
        mt_metrics = dict(ctr_metrics, **mp_metrics)
        tf_summary_metrics(mt_metrics, epoch)
        mt_metrics = dict(mt_metrics, **logs)
        mt_metrics['epoch'] = epoch
        self.result_df = self.result_df.append([mt_metrics], ignore_index=True)

        if not self.light_batch_predict:
            model.Y_pred_train = Y_pred_train
            model.Y_pred_test = Y_pred_test
            model.Z_result_pred_train = Z_result_pred_train
            model.Z_result_pred_test = Z_result_pred_test
            return Y_pred_train, Y_pred_test, Z_result_pred_train, Z_result_pred_test