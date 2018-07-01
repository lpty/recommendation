# coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
from math import exp


class Corpus:

    items_dict_path = 'data/lfm_items.dict'

    @classmethod
    def pre_process(cls):
        file_path = 'data/ratings.csv'
        cls.frame = pd.read_csv(file_path)
        cls.user_ids = set(cls.frame['UserID'].values)
        cls.item_ids = set(cls.frame['MovieID'].values)
        items_dict = {user_id: cls._get_pos_neg_item(user_id) for user_id in list(cls.user_ids)}
        f = open(cls.items_dict_path, 'wb')
        pickle.dump(items_dict, f)
        f.close()

    @classmethod
    def _get_pos_neg_item(cls, user_id):
        """
        Define the pos and neg item for user.
        pos_item mean items that user have rating, and neg_item can be items
        that user never see before.
        Simple down sample method to solve unbalance sample.
        """
        print('Process: {}'.format(user_id))
        pos_item_ids = set(cls.frame[cls.frame['UserID'] == user_id]['MovieID'])
        neg_item_ids = cls.item_ids ^ pos_item_ids
        # neg_item_ids = [(item_id, len(self.frame[self.frame['MovieID'] == item_id]['UserID'])) for item_id in neg_item_ids]
        # neg_item_ids = sorted(neg_item_ids, key=lambda x: x[1], reverse=True)
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict

    @classmethod
    def load(cls):
        f = open(cls.items_dict_path, 'rb')
        items_dict = pickle.load(f)
        f.close()
        return items_dict


class LFM:

    def __init__(self):
        self.class_count = 5
        self.iter_count = 5
        self.alpha = 0.02
        self.lam = 0.01
        self._init_model()

    def _init_model(self):
        """
        Get corpus and initialize model params.
        """
        file_path = 'data/ratings.csv'
        self.frame = pd.read_csv(file_path)
        self.user_ids = set(self.frame['UserID'].values)
        self.item_ids = set(self.frame['MovieID'].values)
        self.items_dict = Corpus.load()

        array_p = np.random.rand(len(self.user_ids), self.class_count)
        array_q = np.random.rand(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def _predict(self, user_id, item_id):
        """
        Calculate interest between user_id and item_id.
        p is the look-up-table for user's interest of each class.
        q means the probability of each item being classified as each class.
        """
        p = np.mat(self.p.ix[user_id].values)
        q = np.mat(self.q.ix[item_id].values).T
        r = (p * q).sum()
        logit = 1.0 / (1 + exp(-r))
        return logit

    def _loss(self, user_id, item_id, y, step):
        """
        Loss Function define as MSE, the code write here not that formula you think.
        """
        e = y - self._predict(user_id, item_id)
        print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.
              format(step, user_id, item_id, y, e))
        return e

    def _optimize(self, user_id, item_id, e):
        """
        Use SGD as optimizer, with L2 p, q square regular.
        e.g: E = 1/2 * (y - predict)^2, predict = matrix_p * matrix_q
             derivation(E, p) = - matrix_q * (y - predict), derivation(E, q) = - matrix_p * (y - predict),
             delta_p = alpha * (derivation(E, p) + l2_square), delta_q = alpha * (derivation(E, q) + l2_square)
        """
        gradient_p = - e * self.q.ix[item_id].values
        l2_p = self.lam * np.square(self.p.ix[user_id].values)
        delta_p = self.alpha * (gradient_p + l2_p)

        gradient_q = - e * self.p.ix[user_id].values
        l2_q = self.lam * np.square(self.q.ix[item_id].values)
        delta_q = self.alpha * (gradient_q + l2_q)

        self.p.loc[user_id] += np.array(delta_p.tolist())
        self.q.loc[item_id] += np.array(delta_q.tolist())

    def train(self):
        """
        Train model.
        """
        for step in range(0, self.iter_count):
            for user_id, item_dict in self.items_dict.items():
                for item_id, y in item_dict.items():
                    e = self._loss(user_id, item_id, y, step)
                    self._optimize(user_id, item_id, e)
            self.alpha *= 0.9
        self.save()

    def predict(self, user_id, top_n=10):
        """
        Calculate all item user have not meet before and return the top n interest items.
        """
        self.load()
        user_item_ids = set(self.frame[self.frame['UserID'] == user_id]['MovieID'])
        other_item_ids = self.item_ids ^ user_item_ids
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def save(self):
        """
        Save model params.
        """
        f = open('data/lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        """
        Load model params.
        """
        f = open('data/lfm.model', 'rb')
        self.p, self.q = pickle.load(f)
        f.close()
