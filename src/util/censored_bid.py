import os
import time
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from config import base_config
from util.processor import Processor, EncodeData
from util.truthful_bidder import TruthfulBidder
import pickle


dataset = 'ipinyou'
campaign_list = base_config.campaign_list[dataset]
# campaign_list = ['2259']
path = base_config.bids_result_path

for camp in campaign_list:
    # load data
    processor = Processor(camp, encode_type='label_encode')
    truthful_bidder = TruthfulBidder()
    truthful_bidder.load_pickle(camp)
    train_encode = processor.load_encode('train')
    X_train = train_encode.X
    B_train = truthful_bidder.bid(X_train)
    m = B_train.shape
    B_train = B_train.reshape((m[0], 1))
    result_path = '{}{}/'.format(path, camp)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    pickle.dump(B_train, open(result_path + 'x_train', 'wb'))