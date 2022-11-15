import sys
sys.path.append('../')
import time
import pickle
import logging
from config import base_config, feature_config
from util.encoder import Encoder
from util.data_loader import DataLoader
from util.data_format import EncodeData


logger = logging.getLogger('tensorflow')


class FeatureProcessor:
    """
    Processor class is used to process data.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = []

    @staticmethod
    def useragent_process(content):
        """
        Process the useragent feature.
        :param content: useragent: str
        :return: 'operation_browser': str
        """
        content = content.lower()
        operation = "other"
        oses = ["windows", "ios", "mac", "android", "linux"]
        browsers = [
            "chrome",
            "sogou",
            "maxthon",
            "safari",
            "firefox",
            "theworld",
            "opera",
            "ie"]
        for o in oses:
            if o in content:
                operation = o
                break
        browser = "other"
        for b in browsers:
            if b in content:
                browser = b
                break
        return operation + "_" + browser

    @staticmethod
    def slot_price_process(content):
        """
        Transform the continuous slot price into several discrete box.
        :param content: slot price: int
        :return: slot price box: str
        """
        price = int(content)
        if price > 100:
            return "101+"
        elif price > 50:
            return "51-100"
        elif price > 10:
            return "11-50"
        elif price > 0:
            return "1-10"
        else:
            return "0"

    def process(self, data, dataset='ipinyou'):
        """
        Process data.
        :param data: original data: Dataframe
        :return: processed data: Dataframe
        """
        print("Start processing data")
        start_time = time.time()
        data = data.fillna("other")
        if self.dataset == 'ipinyou':
            data.loc[:, ['useragent']] = data['useragent'].apply(lambda x: self.useragent_process(x))
            data.loc[:, ["slotprice"]] = data["slotprice"].apply(lambda x: self.slot_price_process(x))
        end_time = time.time()
        print("Process data done. Time cost:{}".format(end_time - start_time))
        return data


class Processor:
    """
    Processor class is used to process data in advance for models.
    """
    def __init__(self, campaign, dataset='ipinyou', encode_type='label_encode'):
        """
        Initilize Preprocessor class with following parameters.
        :param campaign: campaign name: str
        :param dataset: dataset name: str
        :param encode_type: 'not_encode', 'label_encode', "onehot_encode'
        :param train_features: training features: list
        :param categorical_features: categorical features: list
        """
        self.dataset = dataset
        self.campaign = campaign
        self.encode_type = encode_type
        self.X_train = None
        self.Y_train = None
        self.Z_train = None
        self.B_train = None
        self.X_test = None
        self.Y_test = None
        self.Z_test = None
        self.B_test = None
        self.train_features = feature_config.train_features[dataset]
        self.clk_label = base_config.clk_label[dataset]
        self.mp_label = base_config.mp_label[dataset]
        self.bid_label = base_config.bid_label[dataset]
        self.pickle_path = base_config.encode_path + self.encode_type + '/'

    def data_process(self):
        """
        Process data with Processor class.
        :return: None.
        """
        data_loader = DataLoader(self.campaign, dataset=self.dataset)
        # data loading
        train = data_loader.load_data("train")
        test = data_loader.load_data("test")
        # data preprocessing
        feat_processor = FeatureProcessor(dataset=self.dataset)
        train = feat_processor.process(train)
        test = feat_processor.process(test)
        self.X_train = train[self.train_features]
        self.Y_train = train[self.clk_label].values
        self.Z_train = train[self.mp_label].values
        self.B_train = train[self.bid_label].values if self.bid_label else None
        del train
        self.X_test = test[self.train_features]
        self.Y_test = test[self.clk_label].values
        self.Z_test = test[self.mp_label].values
        self.B_test = test[self.bid_label].values if self.bid_label else None
        del test
        return

    def encode(self, save_result=False):
        """
        Encode data with Encode class
        :param save_result: saving result indicicator parameter
        :return: X_train: optional, X_test: optional
        """
        print("start encoding for campaign {}".format(self.campaign))
        self.data_process()
        # data encoding
        encoder = Encoder()
        self.X_train, self.X_test = encoder.encode(self.X_train, self.X_test, self.train_features, self.encode_type)

        if save_result:
            self.dump_pickle('train')
            self.dump_pickle('test')
        return self.X_train, self.X_test

    def dump_pickle(self, data_type):
        """
        Dump encoded data to a pickle file.
        :param data_type: 'train' or 'test'
        :return: None
        """
        pickle_name = '{}_{}.pkl'.format(self.campaign, data_type)
        with open(self.pickle_path+pickle_name, 'wb') as f:
            if data_type == 'train':
                pickle_data = EncodeData(self.X_train, self.Y_train, self.Z_train, self.B_train)
            else:
                pickle_data = EncodeData(self.X_test, self.Y_test, self.Z_test, self.B_test)
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, data_type):
        """
        Load encoded data from the pickle file.
        :param data_type: 'train' or 'test'
        :return: pickle_data: optional
        """
        pickle_name = '{}_{}.pkl'.format(self.campaign, data_type)
        with open(self.pickle_path+pickle_name, 'rb') as f:
            pickle_data = pickle.load(f, encoding='bytes')
        return pickle_data

    def load_encode(self, data_type):
        """
        Load data, including encoded feature X, click label Y and market price Z.
        :param data_type: 'train' or 'test': str
        :param camp: campaign name: str
        :return: encoded data: custom Encode class object (including X, Y, Z attributes)
        """
        logger.info("Start loading {} data for campaign {}".format(data_type, self.campaign))
        start_time = time.time()
        encode = self.load_pickle(data_type)
        encode.X = encode.X[self.train_features]
        end_time = time.time()
        logger.info("Loading data done.Time cost:{}".format(end_time - start_time))
        return encode


if __name__ == '__main__':
    dataset = 'ipinyou'
    # campaign_list = base_config.campaign_list[dataset]
    # take campaign 1458 as an example
    campaign_list = ['1458']
    for camp in campaign_list:
        processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        processor.encode(save_result=True)
        # # load data
        # processor = Processor(campaign=camp, dataset=dataset, encode_type='label_encode')
        # train_encode = processor.load_encode('train')
        # test_encode = processor.load_encode('test')
        # X_train, Y_train, Z_train = train_encode.X, train_encode.Y, train_encode.Z
        # X_test, Y_test, Z_test = test_encode.X, test_encode.Y, test_encode.Z
