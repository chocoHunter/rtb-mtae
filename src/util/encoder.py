import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Encoder:
    """
    Encoder class is used for encoding data.
    """

    def __init__(self):
        """
        Initializing.
        """
        self.encode_features = None

    def encode(self, train, test, features, encode_type):
        """
        Encode data with specific encoding type
        :param train: train data: Dataframe
        :param test: test data: Dataframe
        :param features: encoding features
        :param encode_type: 'not_encode', 'label_encode', 'onehot_encode'
        :return: encoded_train: optional, encoded_test: optional
        """
        self.encode_features = features
        if encode_type == 'not_encode':
            return self.not_encode()
        elif encode_type == 'label_encode':
            return self.label_encode(train, test, features)
        elif encode_type == 'onehot_encode':
            return self.onehot_encode(train, test, features)
        else:
            print("Unknown encode type:{}".format(encode_type))
            return None

    def label_encode(self, train, test, features):
        """
        Label Encoding
        :param train: train data: Dataframe
        :param test: test data: Dataframe
        :param features: encoding features: List
        :return:encoded_train: Dataframe, encoded_test: Dataframe
        """
        start = time.time()
        print("Start label encoding")
        le = LabelEncoder()
        for feat in features:
            train.loc[:, feat] = train.loc[:, feat].astype('str')
            test.loc[:, feat] = test.loc[:, feat].astype('str')
            temp = le.fit_transform(list(train[feat]) + list(test[feat]))
            print("{0}:{1}".format(feat, len(set(temp))))
            train.loc[:, feat] = le.transform(train[feat])
            test.loc[:, feat] = le.transform(test[feat])
        print("{0}:{1}".format("Total_dimen", train.shape))
        print(
            "Data encoding done. Time used:{0:.2f}s".format(
                time.time() - start))
        return train, test

    def onehot_encode(self, train, test, features):
        """
        One hot encoding.
        :param train: train data: Dataframe
        :param test: test data: Dataframe
        :param features: encoding features: List
        :return: encoded_train: sparse.msr_matrix, encoded_test: sparse.msr_matrix
        """
        start = time.time()
        print("Start one hot encoding")
        ohe = OneHotEncoder(categories='auto')
        for i, feat in enumerate(features):
            train[feat] = train[feat].astype('str')
            test[feat] = test[feat].astype('str')
            ohe.fit(np.array(list(train[feat]) +
                             list(test[feat])).reshape(-1, 1))
            temp_train = ohe.transform(train[feat].values.reshape(-1, 1))
            temp_test = ohe.transform(test[feat].values.reshape(-1, 1))
            print("{0} dimensions:{1}".format(feat, temp_train.shape[1]))
            if i == 0:
                encoded_train = temp_train
                encoded_test = temp_test
            else:
                encoded_train = sparse.hstack((encoded_train, temp_train))
                encoded_test = sparse.hstack((encoded_test, temp_test))

        print("Label encoding dimensions:{}".format(encoded_train.shape))
        print("One hot encoding dimensions:{}".format(encoded_train.shape))
        print(
            "Data encoding done. Time used:{0:.2f}s".format(
                time.time() - start))
        return encoded_train, encoded_test
