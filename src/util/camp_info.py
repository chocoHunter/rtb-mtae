# -*- coding: utf-8 -*-
import sys
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(color_codes=True)
from config import base_config


# This class is used to load the basic information of the ad campaign
class CampInfo():
    """
    CampInfo class is used to load the basic information of each ad campaign.
    """
    def __init__(self, camp, data_type="train", dataset='ipinyou'):
        """
        Initialize CampInfo class for a specific campaign.
        :param camp: campaign index
        :param data_type: 'train' or 'test'
        :param dataset: 'ipinyou' or 'yoyi'
        """
        self.camp = camp
        self.data_type = data_type
        self.camp_info_root_path = base_config.camp_info_path
        self.camp_info_path = self.camp_info_root_path + dataset + '/'
        self.data_path = base_config.data_root_path + '{}/{}/'.format(dataset, camp)
        self.cdf_fig_path = self.camp_info_root_path + 'cdf_figure/'
        self.pdf_fig_path = self.camp_info_root_path + 'pdf_figure/'
        self.dataset = dataset

        # the saved pickle path
        self.train_pickle_path = self.camp_info_path + "train_{}.pkl".format(camp)
        self.test_pickle_path = self.camp_info_path + "test_{}.pkl".format(camp)

        # data path
        if self.dataset == 'ipinyou':
            self.train_file = self.data_path + "train.log.txt"
            self.test_file = self.data_path + "test.log.txt"
            self.pay_upper_bound = 300
        elif dataset == 'yoyi':
            self.train_file = self.data_path + "train.yzx.txt"
            self.test_file = self.data_path + "test.yzx.txt"
            self.pay_upper_bound = 1000
        elif dataset == 'criteo':
            self.train_file = self.data_path + "train.log.txt"
            self.test_file = self.data_path + "test.log.txt"
            self.pay_upper_bound = 300

        if data_type == "train":
            self.load_file = self.train_file
            self.pickle_path = self.train_pickle_path
        else:
            self.load_file = self.test_file
            self.pickle_path = self.test_pickle_path

        self.col_name_index = {}
        self.total_imp = 0
        self.total_clk = 0
        self.total_cost = 0
        self.max_pay = 0
        self.min_pay = 0
        self.avg_pay = 0
        self.cpc = 0
        self.cpm = 0
        self.ctr = 0
        self.pay = []
        self.pay_cdf = []
        self.pay_pdf = []
        self.pay_count_dict = {}
        self.pay_pdf_dict = {}
        self.pay_click_dict = {}
        self.pay_ctr_list = []
        for i in range(0, self.pay_upper_bound + 1):
            self.pay_click_dict[i] = 0
            self.pay_count_dict[i] = 0

    def load(self):
        """
        Load original data and perform some basic statistics.
        :return:
        """
        print("Start to load {} data of campaign {}".format(self.data_type, self.camp))
        f_log = open(self.load_file, 'r', encoding='utf-8')
        # read the header and get the information of columns
        if self.dataset == 'ipinyou':
            first_line = f_log.readline()
            columns = first_line.split("\t")
            for index in range(len(columns)):
                col_name = columns[index].strip()
                self.col_name_index[col_name] = index
            pay_index = self.col_name_index["payprice"]
            click_index = self.col_name_index["click"]
        elif self.dataset == 'yoyi':
            click_index = 0
            pay_index = 1
        elif self.dataset == 'criteo':
            first_line = f_log.readline()
            columns = first_line.split("\t")
            for index in range(len(columns)):
                col_name = columns[index].strip()
                self.col_name_index[col_name] = index
            pay_index = self.col_name_index["cost"]
            click_index = self.col_name_index["click"]
        count = 0
        epsilon = sys.float_info.epsilon
        for line in f_log:
            s = line.split("\t")
            pay_price = int(s[pay_index])
            # normalize the pay price of yoyi
            if self.dataset == 'yoyi':
                pay_price = int(pay_price / 1000)
            # outlier
            if pay_price > self.pay_upper_bound:
                pay_price = self.pay_upper_bound
            self.pay.append(pay_price)
            # if click
            if s[click_index] == "1":
                self.total_clk += 1
                self.pay_click_dict[pay_price] += 1

            # count each integer pay price
            self.pay_count_dict[pay_price] += 1
            count += 1
            if count % 1000000 == 0:
                print("Read {} lines".format(count))
        self.total_imp = count
        self.total_cost = sum(self.pay)
        self.max_pay = max(self.pay)
        self.min_pay = min(self.pay)
        self.avg_pay = self.total_cost / self.total_imp

        print(self.total_imp, self.total_clk)
        self.cpc = self.total_cost / self.total_clk
        self.cpm = self.total_cost / self.total_imp * 1000
        self.ctr = self.total_clk / self.total_imp

        for i in range(0, self.pay_upper_bound + 1):
            self.pay_pdf_dict[i] = self.pay_count_dict[i] / self.total_imp
            pay_click = self.pay_click_dict[i]
            pay_count = self.pay_count_dict[i]
            self.pay_ctr_list.append(float(pay_click / (pay_count + epsilon)))

        last_pay_cdf = 0
        for x in range(0, self.pay_upper_bound + 1):
            curr_pay_pdf = self.pay_pdf_dict.get(x, 0)
            self.pay_pdf.append(curr_pay_pdf)
            self.pay_cdf.append(last_pay_cdf + curr_pay_pdf)
            last_pay_cdf = self.pay_cdf[-1]

        print("Load {} data of campaign {} done!".format(self.data_type, self.camp))
        print("Total impressions:{}".format(self.total_imp))
        print("Total clicks:{}".format(self.total_clk))

    def dump_pickle(self):
        """
        Dump the campaign information statistics to the pickle file.
        :return: None
        """
        if not os.path.exists(self.camp_info_path):
            os.mkdir(self.camp_info_path)
        pickle.dump(self, open(self.pickle_path, 'wb'))

    # load pickle and return CampInfo class data
    def load_pickle(self):
        """
        Load the campaign information statistics from the pre-saved pickle file.
        :return: None
        """
        with open(self.pickle_path, 'rb') as f:
            pickle_info = pickle.load(f, encoding='bytes')
            for key, value in vars(pickle_info).items():
                setattr(self, key, value)

    def plot_cdf(self, save_fig=True):
        """
        To plot the cdf of the market price for the specific campaign, you need to load the data in advance.
        :param save_fig: Saving figure indicator parameter. Default true.
        :return: None
        """
        if len(self.pay_cdf) != 0:
            plt.rcParams.update({'font.size': 18})
            plt.plot(self.pay_cdf, '-', label='Truth cdf')
            plt.ylabel('winning probability')
            plt.xlabel('market price'.format(self.camp, self.data_type))
            # plt.title('the cdf of marker price (campaign {} {})'.format(self.camp, self.data_type))
            if save_fig:
                plt.savefig('{}{}_{}_{}_cdf.pdf'.format(self.cdf_fig_path, self.dataset, self.camp, self.data_type), bbox_inches='tight')
            plt.show()
        else:
            print("The cdf of market price is none. Please first load the data and then plot cdf")

    def plot_pdf(self, save_fig=True):
        """
        To plot the pdf of the market price for the specific campaign, you need to load the data in advance.
        :param save_fig: Saving figure indicator parameter. Default true.
        :return: None
        """
        if len(self.pay_pdf) != 0:
            plt.hist(self.pay, bins=20, histtype="stepfilled",normed=True,alpha=0.6)
            plt.ylabel('probability')
            plt.xlabel('market price')
            sns.kdeplot(self.pay, shade=True)
            # plt.title('the pdf of marker price (campaign {} {})'.format(self.camp, self.data_type))
            if save_fig:
                plt.savefig('{}{}_{}_{}_pdf.pdf'.format(self.pdf_fig_path, self.dataset, self.camp, self.data_type), bbox_inches='tight')
            plt.show()
        else:
            print("The pdf of market price is none. Please first load the data and then plot pdf")


if __name__ == '__main__':
    dataset = 'ipinyou'
    campaign_list = base_config.campaign_list[dataset]
    campaign_list = ['2259']
    type_list = ['train', 'test']

    camp_base_info_dict = {
        'dataset': [],
        'camp': [],
        'data_type': [],
        'total_imp': [],
        'total_clk': [],
        'total_cost': [],
        'ctr': [],
        'cpm': [],
        'cpc': [],
        'avg_pay': [],
        'max_pay': [],
        'min_pay': []
    }
    camp_cpc_dict = {}
    for data_type in type_list:
        for camp in campaign_list:
            camp_info = CampInfo(camp, data_type=data_type, dataset=dataset)
            # camp_info.load()
            # camp_info.plot_cdf()
            # camp_info.plot_pdf()
            # camp_info.dump_pickle()
            # print("load and dump_pickle " + camp + " done!")

            # for i in camp_base_info_dict.keys():
            #     camp_base_info_dict[i].append(camp_info.__getattribute__(i))
            #
            camp_info.load_pickle()
            camp_cpc_dict[camp] = round(camp_info.cpc)
            print('{}:{}'.format(data_type, round(camp_info.cpc)))
    #
    # camp_info_df = pd.DataFrame.from_dict(camp_base_info_dict)
    # camp_info_df.to_excel(base_config.camp_info_path + 'criteo_camp_info.xlsx', index=False)