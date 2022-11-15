import sys
sys.path.append('../')
import numpy as np
import logging
from config import base_config
from util.processor import Processor, EncodeData
from util.truthful_bidder import TruthfulBidder
from util.data_format import CensoredBatchData

np.random.seed(2020)
logger = logging.getLogger('tensorflow')


class CensoredProcessor:
    def __init__(self, processor, bidder):
        self.processor = processor
        self.bidder = bidder
        self.win_data = None
        self.lose_data = None

    def load_encode(self, data_type='train', load_bid=True):
        data = self.processor.load_encode(data_type)
        if load_bid:
            bids = self.bidder.load_bids()
        else:
            bids = self.bidder.bid(data.X)
        win_index = data.Z < bids
        lose_index = data.Z >= bids
        win_data = EncodeData(X=data.X[win_index], Y=data.Y[win_index],
                              Z=data.Z[win_index], B=bids[win_index])
        lose_data = EncodeData(X=data.X[lose_index], Y=data.Y[lose_index],
                               Z=data.Z[lose_index], B=bids[lose_index])
        return win_data, lose_data

    def generate_batch_index(self, size, batch_size=10240):
        start_indexs = [i * batch_size for i in range(int(np.ceil(size / batch_size)))]
        end_indexs = [i * batch_size for i in range(1, int(np.ceil(size / batch_size)))]
        end_indexs.append(size)
        return start_indexs, end_indexs

    def load_dataset(self, batch_size=10240, data_type='train'):
        win_data, lose_data = self.load_encode(data_type=data_type)
        win_size, lose_size = len(win_data.Z), len(lose_data.Z)
        win_start_indexs, win_end_indexs = self.generate_batch_index(win_size, batch_size)
        lose_start_indexs, lose_end_indexs = self.generate_batch_index(lose_size, batch_size)
        censored_win_dataset = self.generate_censored_batches(win_data, win_start_indexs, win_end_indexs, True)
        censored_lose_dataset = self.generate_censored_batches(lose_data, lose_start_indexs, lose_end_indexs, False)
        censored_full_dataset = censored_win_dataset + censored_lose_dataset
        np.random.shuffle(censored_full_dataset)
        for i in range(len(censored_full_dataset)):
            censored_full_dataset[i].batch_id = i
        return censored_full_dataset

    def generate_censored_batches(self, data, start_indexs, end_indexs, win_flag):
        censored_batch_dataset = []
        for i in range(len(start_indexs)):
            start = start_indexs[i]
            end = end_indexs[i]
            batch_data = self.data_slice(data, start, end)
            censored_batch_data = CensoredBatchData(batch_data, batch_id=-1, win_flag=win_flag)
            censored_batch_dataset.append(censored_batch_data)
        return censored_batch_dataset

    @staticmethod
    def data_slice(data, start_index, end_index):
        return EncodeData(data.X[start_index:end_index],
                          data.Y[start_index:end_index],
                          data.Z[start_index:end_index],
                          data.B[start_index:end_index])


if __name__ == '__main__':
    dataset = 'ipinyou'
    # campaign_list = base_config.campaign_list[dataset]
    # take campaign 1458 as an example
    campaign_list = ['1458']
    for camp in campaign_list:
        # load data
        processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        clk_value = base_config.camp_cpc_dict[dataset][camp]
        truthful_bidder = TruthfulBidder(dataset=dataset, campaign=camp)
        censored_processor = CensoredProcessor(processor, truthful_bidder)
        censored_dataset = censored_processor.load_dataset(batch_size=10240)
        win_train, lose_train = censored_processor.load_encode('train')
