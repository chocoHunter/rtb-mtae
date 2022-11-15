# Root path
root_path = '/home/rtb-mtae/'

# Data source path
src_path = root_path + 'src/'
dataset = 'ipinyou'
data_root_path = root_path + 'data/'
ipinyou_data_path = data_root_path + 'ipinyou/'
criteo_data_path = data_root_path + 'criteo/'
yoyi_data_path = data_root_path + 'yoyi/'

# Default directory path
encode_path = root_path + "encode/"
bidder_path = root_path + 'bidder/'
bids_result_path = bidder_path + 'bids_result/'
camp_info_path = root_path + 'camp_info/'
result_path = root_path + 'result/'
log_path = root_path + 'log/'
figure_path = root_path + 'figure/'
model_path = root_path + 'model/'
model_tb_path = root_path + 'model/tensorboard'
model_cp_path = root_path + 'model/checkpoint'
model_predict_path = root_path + 'model/predict'

# campaign list
ipinyou_campaign_list = [
    '1458',
    '2259',
    '2261',
    '2821',
    '2997',
    '3358',
    '3386',
    '3427',
    '3476',
    'all']
criteo_campaign_list = ['criteo']
yoyi_campaign_list = ['sample']
campaign_list = {
    'ipinyou': ipinyou_campaign_list,
    'yoyi': yoyi_campaign_list,
    'criteo': criteo_campaign_list
}

data_path = {
    'ipinyou': ipinyou_data_path,
    'yoyi': yoyi_data_path,
    'criteo': criteo_data_path
}

# the cpc dictionary of campaigns
camp_cpc_dict = {'ipinyou': {'1458': 86553,
                             '2259': 277696,
                             '2261': 297637,
                             '2821': 140074,
                             '2997': 14206,
                             '3358': 118515,
                             '3386': 105524,
                             '3427': 109159,
                             '3476': 151985,
                             'all': 106937},
                 'yoyi': {'sample': 998},
                 'criteo': {'criteo': 63}
                 }

clk_label = {'ipinyou': 'click',
             'yoyi': 'click',
             'criteo': 'click'}

mp_label = {'ipinyou': 'payprice',
             'yoyi': 'payprice',
             'criteo': 'cost'}

bid_label = {'ipinyou': 'bidprice',
             'yoyi': 'bidprice',
             'criteo': None}

click_value = 100000
max_payprice = 300
max_bidprice = 300