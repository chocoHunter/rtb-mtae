# feature sizes
feature_nums = {
    'ipinyou':{
        '1458':560870,
        '2259':97500,
        '2261':333223,
        '2821':460925,
        '2997':133541,
        '3358':491768,
        '3386':556952,
        '3427':551226,
        '3476':490726,
        'all':937750
    },
    'yoyi':{
        'sample':3036119
    }
}

# feature
common_features = [
    "weekday",
    "hour",
    "IP",
    "region",
    "city",
    "adexchange",
    "domain",
    "slotid",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "creative",
    "advertiser",
    "useragent",
    "slotprice",
    # 'payprice'
    # 'imp_freq',
    # 'clk_freq',
    # 'is_imp_before',
    # 'is_clk_before',
    # "last_imp_time_interval",
    # "last_clk_time_interval"
]

ipinyou_categorical_features = [
    "weekday",
    "hour",
    # "IP",
    "region",
    "city",
    "adexchange",
    "domain",
    "slotid",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "creative",
    "advertiser",
    "useragent",
    "slotprice"]

ipinyou_usertag_features = ['usertag_10006',
                    'usertag_10024',
                    'usertag_10031',
                    'usertag_10048',
                    'usertag_10052',
                    'usertag_10057',
                    'usertag_10059',
                    'usertag_10063',
                    'usertag_10067',
                    'usertag_10074',
                    'usertag_10075',
                    'usertag_10076',
                    'usertag_10077',
                    'usertag_10079',
                    'usertag_10083',
                    'usertag_10093',
                    'usertag_10102',
                    'usertag_10110',
                    'usertag_10111',
                    'usertag_10114',
                    'usertag_10115',
                    'usertag_10116',
                    'usertag_10117',
                    'usertag_10118',
                    'usertag_10120',
                    'usertag_10123',
                    'usertag_10125',
                    'usertag_10126',
                    'usertag_10127',
                    'usertag_10129',
                    'usertag_10130',
                    'usertag_10131',
                    'usertag_10133',
                    'usertag_10138',
                    'usertag_10140',
                    'usertag_10142',
                    'usertag_10145',
                    'usertag_10146',
                    'usertag_10147',
                    'usertag_10148',
                    'usertag_10149',
                    'usertag_10684',
                    'usertag_11092',
                    'usertag_11278',
                    'usertag_11379',
                    'usertag_11423',
                    'usertag_11512',
                    'usertag_11576',
                    'usertag_11632',
                    'usertag_11680',
                    'usertag_11724',
                    'usertag_11944',
                    'usertag_13042',
                    'usertag_13403',
                    'usertag_13496',
                    'usertag_13678',
                    'usertag_13776',
                    'usertag_13800',
                    'usertag_13866',
                    'usertag_13874',
                    'usertag_14273',
                    'usertag_15398',
                    'usertag_16593',
                    'usertag_16617',
                    'usertag_16661',
                    'usertag_16706',
                    'usertag_16751',
                    'usertag_16753']

ipinyou_train_features = ipinyou_categorical_features + ipinyou_usertag_features

criteo_info_features = ['cost', 'cpo', 'time_since_last_click']
criteo_train_features = ['campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6',
            'cat7', 'cat8']

train_features = {
    'ipinyou': ipinyou_train_features,
    'criteo': criteo_train_features
}