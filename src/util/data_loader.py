import time
import pandas as pd
from config import base_config


class DataLoader:
    """
    DataLoader class is used to load the pre-processed data as a dataframe type.
    """
    def __init__(self, campaign, dataset='ipinyou'):
        """
        Initialize DataLoader class for a specific campaign
        :param campaign:
        :param dataset:
        """
        self.campaign = campaign
        self.dataset = dataset
        self.data_path = base_config.data_path[dataset]

    def load_data(self, data_type):
        """
        Load data with data_type parameter.
        :param data_type: 'train' or 'test'
        :return: data: Dataframe type
        """
        if (data_type != "train") and (data_type != "test"):
            print("data_type unknown. Please input train or test.")
            return None
        print("Start loading {} data of campaign {}".format(data_type, self.campaign))
        start_time = time.time()
        data = pd.read_csv("{}/{}.log.txt".format(self.data_path+self.campaign, data_type), sep='\t')
        end_time = time.time()
        print("Loading {} data done.Time cost:{}".format(data_type, end_time - start_time))
        return data


if __name__ == '__main__':
    campaign_list = base_config.campaign_list
    campaign_list = ['2259']
    for campaign in campaign_list:
        data_load = DataLoader(campaign)
        train = data_load.load_data('train')