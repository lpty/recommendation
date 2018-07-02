# coding: utf-8 -*-
import pickle
import pandas as pd


class Graph:

    graph_path = 'data/prank.graph'

    @classmethod
    def _gen_user_graph(cls, user_id):
        print('Gen graph user: {}'.format(user_id))
        item_ids = list(set(cls.frame[cls.frame['UserID'] == user_id]['MovieID']))
        graph_dict = {'item_{}'.format(item_id): 1 for item_id in item_ids}
        return graph_dict

    @classmethod
    def _gen_item_graph(cls, item_id):
        print('Gen graph item: {}'.format(item_id))
        user_ids = list(set(cls.frame[cls.frame['MovieID'] == item_id]['UserID']))
        graph_dict = {'user_{}'.format(user_id): 1 for user_id in user_ids}
        return graph_dict

    @classmethod
    def gen_graph(cls):
        """
        Gen graph.Each user,movie define as a node, and every movie rated by user means
        that there is a edge between user and movie, edge weight is 1 simply.
        """
        file_path = 'data/ratings.csv'
        cls.frame = pd.read_csv(file_path)
        user_ids = list(set(cls.frame['UserID']))
        item_ids = list(set(cls.frame['MovieID']))
        cls.graph = {'user_{}'.format(user_id): cls._gen_user_graph(user_id) for user_id in user_ids}
        for item_id in item_ids:
            cls.graph['item_{}'.format(item_id)] = cls._gen_item_graph(item_id)
        cls.save()

    @classmethod
    def save(cls):
        f = open(cls.graph_path, 'wb')
        pickle.dump(cls.graph, f)
        f.close()

    @classmethod
    def load(cls):
        f = open(cls.graph_path, 'rb')
        graph = pickle.load(f)
        f.close()
        return graph


class PersonalRank:

    def __init__(self):
        self.graph = Graph.load()
        self.alpha = 0.6
        self.iter_count = 20
        self._init_model()

    def _init_model(self):
        """
        Initialize prob of every node, zero default.
        """
        self.params = {k: 0 for k in self.graph.keys()}

    def train(self, user_id):
        """
        For target user, every round will start at that node, means prob will be 1.
        And node will be updated by formula like:
        for each node, if node j have edge between i:
            prob_i_j = alpha * prob_j / edge_num_out_of_node_j
            then prob_i += prob_i_j
        alpha means the prob of continue walk.
        """
        self.params['user_{}'.format(user_id)] = 1
        for count in range(self.iter_count):
            print('Step {}...'.format(count))
            tmp = {k: 0 for k in self.graph.keys()}
            # edges mean all edge out of node
            for node, edges in self.graph.items():
                for next_node, _ in edges.items():
                    # every edge come in next_node update prob
                    tmp[next_node] += self.alpha * self.params[node] / len(edges)
            # root node.
            tmp['user_' + str(user_id)] += 1 - self.alpha
            self.params = tmp
        self.params = sorted(self.params.items(), key=lambda x: x[1], reverse=True)
        self.save(user_id)

    def predict(self, user_id, top_n=10):
        """
        Return top n node without movie target user have been rated and other user.
        """
        self.load(user_id)
        frame = pd.read_csv('data/ratings.csv')
        item_ids = ['item_' + str(item_id) for item_id in list(set(frame[frame['UserID'] == user_id]['MovieID']))]
        candidates = [(key, value) for key, value in self.params if key not in item_ids and 'user' not in key]
        return candidates[:top_n]

    def save(self, user_id):
        f = open('data/prank_{}.model'.format(user_id), 'wb')
        pickle.dump(self.params, f)
        f.close()

    def load(self, user_id):
        f = open('data/prank_{}.model'.format(user_id), 'rb')
        self.params = pickle.load(f)
        f.close()
