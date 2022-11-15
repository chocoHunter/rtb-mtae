import os
import time
import sys
sys.path.append('../')
import logging
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from config import base_config
from util.processor import Processor, EncodeData
from util.censored_prosessor import CensoredProcessor
from util.truthful_bidder import TruthfulBidder
from common.callbacks import LogMTLossCallback
from mt_model.mtae_model import MTAE


np.random.seed(2020)
tf.random.set_seed(2020)

dataset = 'ipinyou'
camp = sys.argv[1] if len(sys.argv) >= 2 else '1458'
CUDA_VISIBLE_DEVICES = sys.argv[2] if len(sys.argv) >= 3 else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# hyperparameter
TRAIN_BATCH_SIZE = 10240
PREDICT_BATCH_SIZE = 102400
LEARNING_RATE = 5e-5
EPOCH = 30
PREDICT_Z_SIZE = 301
MP_LOSS_WEIGHT = 0.01
CTR_LOSS_WEIGHT = 1 - MP_LOSS_WEIGHT
ANLP_LOSS_WEIGHT = 1 * MP_LOSS_WEIGHT
WIN_RATE_LOSS_WEIGHT = 1e-6
LOSE_RATE_LOSS_WEIGHT = 1e-6

# data settings
data_path = base_config.data_root_path
output_dir_name = 'mt_model/mtae_ctr{}_mp{}_anlp{}_winr{}_lose{}/{}_{}_{}'.format(CTR_LOSS_WEIGHT, MP_LOSS_WEIGHT, ANLP_LOSS_WEIGHT, WIN_RATE_LOSS_WEIGHT, LOSE_RATE_LOSS_WEIGHT, TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCH)

# logger
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)

# log redirection
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = base_config.log_path + output_dir_name
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file = '{}/{}_{}_{}'.format(log_path, dataset, camp, log_time)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


result_path = base_config.log_path + '{}/'.format(output_dir_name)
figure_path = result_path + 'figure/{}/'.format(camp)
pdf_path = result_path + 'pdf/{}/'.format(camp)


# load data
processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
train_data = processor.load_encode('train')
test_data = processor.load_encode('test')
X_train, Y_train, Z_train = train_data.X, train_data.Y, train_data.Z
X_test, Y_test, Z_test = test_data.X, test_data.Y, test_data.Z

truthful_bidder = TruthfulBidder(dataset=dataset, campaign=camp)
censored_processor = CensoredProcessor(processor, truthful_bidder)
censored_dataset = censored_processor.load_dataset(TRAIN_BATCH_SIZE, 'train')

# feature vocabulary
combine_data = pd.concat([X_train, X_test])
feature_vocab = dict(combine_data.nunique().items())
del combine_data

# transform to (feature_size, data_size) as input
X_train = [X_train.values[:, k] for k in range(X_train.values.shape[1])]
X_test = [X_test.values[:, k] for k in range(X_test.values.shape[1])]

log_cb = LogMTLossCallback(X_train, X_test, Y_train, Y_test, Z_train, Z_test, predict_batch_size=PREDICT_BATCH_SIZE, predict_type='softmax', predict_z_size=PREDICT_Z_SIZE,
                           save_fig=True, figure_path=figure_path, pdf_path=pdf_path, camp=camp, light_batch_predict=False)

mtae = MTAE(feature_vocab=feature_vocab,
            LEARNING_RATE=LEARNING_RATE,
            EPOCH=EPOCH,
            TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE,
            PREDICT_Z_SIZE=PREDICT_Z_SIZE,
            CTR_LOSS_WEIGHT=CTR_LOSS_WEIGHT,
            MP_LOSS_WEIGHT=MP_LOSS_WEIGHT,
            ANLP_LOSS_WEIGHT=ANLP_LOSS_WEIGHT,
            WIN_RATE_LOSS_WEIGHT=WIN_RATE_LOSS_WEIGHT,
            LOSE_RATE_LOSS_WEIGHT=LOSE_RATE_LOSS_WEIGHT,
            log_cb=log_cb)
mtae.full_model.summary(print_fn=logger.info)

# start training
train_start_time = time.time()
logger.info('Start training. Time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start_time))))

mtae.fit(censored_dataset)

train_end_time = time.time()
logger.info('End training. Time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_end_time))))
logger.info("Time cost of training: {:.0f}s".format(train_end_time-train_start_time))

# save the log results as xlsx file
log_cb.result_df.to_excel(result_path + '{}.xlsx'.format(camp))