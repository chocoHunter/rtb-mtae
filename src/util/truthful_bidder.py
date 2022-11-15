import os
import pickle
import sys
sys.path.append('../')
import numpy as np
from ctr_baseline.lgb import LgbCtrPredictor
from config import base_config, feature_config
from util.processor import Processor
from util.data_format import EncodeData


class TruthfulBidder():
    """
    TruthfulBidder class is used to apply truthful bidding.
    """
    def __init__(self, campaign='2259', dataset='ipinyou', ctr_predictor=LgbCtrPredictor(), clip_bid_price=300):
        """
        Initialize the TruthfulBidder class.
        :param clk_value: specific value for a click. (default: cpc)
        :param ctr_predictor: CTR predict model. (censor division model: lightgbm: LgbCtrPredictor)
        """
        self.ctr_predictor = ctr_predictor
        self.campaign = campaign
        self.clk_value = base_config.camp_cpc_dict[dataset][campaign]
        self.clip_bid_price = clip_bid_price
        self.pickle_path = base_config.bidder_path + 'truthful_bidder/'

    def bid(self, X):
        """
        Apply the truthful bidding: b = ctr * clk_value
        :param X: label_encode feature X: Dataframe
        :return: bid price: np.array shape=(len(X),)
        """
        ctr = self.ctr_predictor.model.predict_proba(X, )[:, 1]
        bids = np.ceil(ctr * self.clk_value)
        if self.clip_bid_price:
            bids = np.clip(bids, 0, self.clip_bid_price)
        return bids

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.ctr_predictor.fit(X_train, Y_train, X_test, Y_test)

    def dump_pickle(self):
        pickle_name = '{}.pkl'.format(self.campaign)
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        with open(self.pickle_path + pickle_name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self):
        pickle_name = '{}.pkl'.format(self.campaign)
        with open(self.pickle_path+pickle_name, 'rb') as f:
            pickle_bidder = pickle.load(f, encoding='bytes')
            for name, val in vars(pickle_bidder).items():
                self.__setattr__(name, val)

    def dump_bids(self, X):
        bids = self.bid(X)
        pickle_name = '{}_B_train.pkl'.format(self.campaign)
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        with open(self.pickle_path + pickle_name, 'wb') as f:
            pickle.dump(bids, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_bids(self):
        pickle_name = '{}_B_train.pkl'.format(self.campaign)
        with open(self.pickle_path + pickle_name, 'rb') as f:
            bids = pickle.load(f, encoding='bytes')
        return bids


if __name__ == '__main__':
    dataset = 'ipinyou'
    # campaign_list = base_config.campaign_list[dataset]
    # take campaign 1458 as an example
    campaign_list = ['1458']
    for camp in campaign_list:
        # load data
        print('camp:{}'.format(camp))
        processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        train_encode = processor.load_encode('train')
        test_encode = processor.load_encode('test')
        X_train, Y_train = train_encode.X, train_encode.Y
        X_test, Y_test = test_encode.X, test_encode.Y

        truthful_bidder = TruthfulBidder(campaign=camp, dataset=dataset, clip_bid_price=300)
        truthful_bidder.fit(X_train, Y_train, X_test, Y_test)
        truthful_bidder.dump_bids(X_train)
        # truthful_bidder.load_pickle()
        # bid_list_test = truthful_bidder.bid(X_test)
        # truthful_bidder.dump_pickle(camp)
        # test_bidder = TruthfulBidder()
        # test_bidder.load_pickle(camp)