import os
import time
import sys
sys.path.append('../')
import logging
import warnings
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# logger
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)


class MTAE():
    def __init__(self, feature_vocab, LEARNING_RATE=0.0005, EPOCH=20, TRAIN_BATCH_SIZE=10240, PREDICT_Z_SIZE=301, CTR_LOSS_WEIGHT=0.5,
                 MP_LOSS_WEIGHT=0.5, ANLP_LOSS_WEIGHT=0.2, WIN_RATE_LOSS_WEIGHT=0.8, LOSE_RATE_LOSS_WEIGHT=0.8, log_cb=None):
        self.feature_vocab = feature_vocab
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCH = EPOCH
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.PREDICT_Z_SIZE = PREDICT_Z_SIZE
        self.CTR_LOSS_WEIGHT = CTR_LOSS_WEIGHT
        self.MP_LOSS_WEIGHT = MP_LOSS_WEIGHT
        self.ANLP_LOSS_WEIGHT = ANLP_LOSS_WEIGHT
        self.WIN_RATE_LOSS_WEIGHT = WIN_RATE_LOSS_WEIGHT
        self.LOSE_RATE_LOSS_WEIGHT = LOSE_RATE_LOSS_WEIGHT
        self.log_cb = log_cb
        self.full_model, self.ctr_anlp_model, self.win_rate_model = self.generate_model(feature_vocab=self.feature_vocab,
                                                                                        PREDICT_Z_SIZE=self.PREDICT_Z_SIZE,
                                                                                        ANLP_LOSS_WEIGHT=self.ANLP_LOSS_WEIGHT,
                                                                                        WIN_RATE_LOSS_WEIGHT=self.WIN_RATE_LOSS_WEIGHT,
                                                                                        LOSE_RATE_LOSS_WEIGHT=self.LOSE_RATE_LOSS_WEIGHT)
        self.log_cb.import_model = self.ctr_anlp_model

    def generate_model(self, feature_vocab, PREDICT_Z_SIZE=301, CTR_LOSS_WEIGHT=0.5, ANLP_LOSS_WEIGHT=0.5,
                       WIN_RATE_LOSS_WEIGHT=1e-5, LOSE_RATE_LOSS_WEIGHT=1e-5):
        """
        Generate the model for ctr predicting
        :param feature_vocab: vocabulary of each features: dict
        :return: CTR model:  tensorflow.python.keras.engine.training.Model
        """
        models = []
        inputs = []
        # FEM
        # feature-wise embedding
        for feature in feature_vocab.keys():
            vocab = feature_vocab[feature]
            embed_dim = int(min(np.ceil(vocab / 2), 50))
            input_layer = layers.Input(shape=(1,),
                                                name='input_' + '_'.join(feature.split(' ')))
            embed_layer = layers.Embedding(vocab, embed_dim, trainable=True,
                                           embeddings_initializer=tf.keras.initializers.he_normal())(input_layer)
            embed_reshape_layer = layers.Reshape(target_shape=(embed_dim,))(embed_layer)
            models.append(embed_reshape_layer)
            inputs.append(input_layer)

        input_merge = layers.concatenate(models, name='input_merge')
        # SRM
        share_hidden1 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='share_hidden1')(
            input_merge)
        share_hidden2 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='share_hidden2')(
            share_hidden1)
        share_hidden3 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='share_hidden3')(
            share_hidden2)
        # CPMM
        ctr_hideen1 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='ctr_hideen1')(
            share_hidden3)
        ctr_hideen2 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='ctr_hideen2')(
            ctr_hideen1)
        ctr_hideen3 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='ctr_hideen3')(
            ctr_hideen2)
        # MPMM
        mp_hidden1 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='mp_hidden_1')(
            share_hidden3)
        mp_hidden2 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='mp_hidden_2')(
            mp_hidden1)
        mp_hidden3 = layers.Dense(300, activation='selu', kernel_initializer='lecun_normal', name='mp_hidden_3')(
            mp_hidden2)

        ctr_output = layers.Dense(1, activation='sigmoid', name='ctr_output')(ctr_hideen3)

        mp_output = layers.Dense(PREDICT_Z_SIZE, activation='softmax', name='mp_output')(mp_hidden3)

        # def reduce_cdf(pdf):
        #     def reduce_sum(x):
        #         cdf = [x[0]]
        #         for i in range(1, x.shape[-1]):
        #             cdf.append(tf.add(cdf[i-1], x[i]))
        #         return tf.stack(cdf)
        #     cdf = tf.vectorized_map(reduce_sum, elems=pdf)
        #     return cdf

        # def win_rate(inputs):
        #     cdf = inputs[0]
        #     bid_price = tf.reshape(tf.cast(inputs[1], dtype=tf.int32), shape=(-1,1))
        #     line = tf.reshape(tf.range(tf.shape(bid_price)[0]), shape=(-1, 1))
        #     bid_index = tf.concat([line, bid_price], axis=1)
        #     bid_win_rate = tf.gather_nd(cdf, bid_index)
        #     return bid_win_rate

        # WPPM
        bid_price_input = layers.Input(shape=(1,), name='bid_price_input')
        mp_cdf = layers.Lambda(self.reduce_cdf)(mp_output)
        win_rate_output = layers.Lambda(self.win_rate)([mp_cdf, bid_price_input])

        # MTAE
        full_model = tf.keras.models.Model(inputs=[inputs, bid_price_input], outputs=[ctr_output, mp_output, win_rate_output])
        full_model.compile(loss=[tf.keras.losses.binary_crossentropy, tf.keras.losses.sparse_categorical_crossentropy, tf.keras.losses.binary_crossentropy],
                           loss_weights=[CTR_LOSS_WEIGHT, ANLP_LOSS_WEIGHT, WIN_RATE_LOSS_WEIGHT],
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))
        ctr_anlp_model = tf.keras.models.Model(inputs=[inputs], outputs=[ctr_output, mp_output])
        win_rate_model = tf.keras.models.Model(inputs=[inputs, bid_price_input], outputs=[win_rate_output])
        win_rate_model.compile(loss=[tf.keras.losses.binary_crossentropy],
                               loss_weights=[LOSE_RATE_LOSS_WEIGHT],
                               optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))
        return full_model, ctr_anlp_model, win_rate_model

    @staticmethod
    def reduce_cdf(pdf):
        def reduce_sum(x):
            cdf = [x[0]]
            for i in range(1, x.shape[-1]):
                cdf.append(tf.add(cdf[i - 1], x[i]))
            return tf.stack(cdf)

        cdf = tf.vectorized_map(reduce_sum, elems=pdf)
        return cdf

    @staticmethod
    def win_rate(inputs):
            cdf = inputs[0]
            bid_price = tf.reshape(tf.cast(inputs[1], dtype=tf.int32), shape=(-1,1))
            line = tf.reshape(tf.range(tf.shape(bid_price)[0]), shape=(-1, 1))
            bid_index = tf.concat([line, bid_price], axis=1)
            bid_win_rate = tf.gather_nd(cdf, bid_index)
            return bid_win_rate

    def fit(self, censored_dataset):
        for epoch in range(self.EPOCH):
            print("Epoch {}/{}".format(epoch + 1, self.EPOCH))
            batch_size = len(censored_dataset)
            ctr_cross_entropy_loss = []
            anlp_loss = []
            win_cross_entropy_loss = []
            lose_cross_entropy_loss = []
            logs = defaultdict(list)
            for step, batch_data in enumerate(censored_dataset):
                if batch_data.win_flag:
                    batch_X, batch_Y, batch_Z, batch_B = batch_data.data.X, batch_data.data.Y, batch_data.data.Z, batch_data.data.B
                    batch_X = [batch_X.values[:, k] for k in range(batch_X.values.shape[1])]
                    batch_win = np.ones(batch_data.size)
                    win_loss = self.full_model.train_on_batch(x=[batch_X, batch_B],
                                                   y=[batch_Y, batch_Z, batch_win])
                    win_cross_entropy_loss.append(win_loss[3])
                    ctr_cross_entropy_loss.append(win_loss[1])
                    anlp_loss.append(win_loss[2])
                else:
                    batch_X, batch_B = batch_data.data.X, batch_data.data.B
                    batch_X = [batch_X.values[:, k] for k in range(batch_X.values.shape[1])]
                    batch_win = np.zeros(batch_data.size)
                    lose_loss = self.win_rate_model.train_on_batch(x=[batch_X, batch_B],
                                                       y=[batch_win])
                    lose_cross_entropy_loss.append(lose_loss/self.WIN_RATE_LOSS_WEIGHT)

                # ignore the warning of empty slice for np.mean()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_ctr_cross_entropy_loss = np.mean(ctr_cross_entropy_loss)
                    mean_anlp_loss = np.mean(anlp_loss)
                    mean_win_cross_entropy_loss = np.mean(win_cross_entropy_loss)
                    mean_lose_cross_entropy_loss = np.mean(lose_cross_entropy_loss)
                    mean_total_cross_entropy_loss = np.mean(win_cross_entropy_loss + lose_cross_entropy_loss)
                print("{:.2%}  Step[{}/{}]. Win flag: {}. ctr_ce: {:.5f} anlp: {:.5f} win_ce: {:.5f} lose_ce: {:.5f} total_ce:{:.5f}".format((step + 1) / batch_size, step + 1, batch_size, batch_data.win_flag,
                                                                             mean_ctr_cross_entropy_loss, mean_anlp_loss,
                                                                             mean_win_cross_entropy_loss, mean_lose_cross_entropy_loss, mean_total_cross_entropy_loss), end='\r')
            logs['train_ctr_cross_entropy'].append(mean_ctr_cross_entropy_loss)
            logs['train_mean_anlp_loss'].append(mean_anlp_loss)
            logs['train_win_cross_entropy'].append(mean_win_cross_entropy_loss)
            logs['train_lose_cross_entropy'].append(mean_lose_cross_entropy_loss)
            logs['train_total_cross_entropy'].append(mean_total_cross_entropy_loss)
            logger.info('ctr_ce: {:.5f} anlp: {:.5f} win_ce: {:.5f} lose_ce: {:.5f} total_ce:{:.5f}'.format(mean_ctr_cross_entropy_loss, mean_anlp_loss,
                                                                              mean_win_cross_entropy_loss, mean_lose_cross_entropy_loss, mean_total_cross_entropy_loss))

            self.log_cb.on_epoch_end(epoch+1, logs)
