# -*- coding: utf-8 -*-
import time
import os
from model.prank import Graph, PersonalRank


def run():
    assert os.path.exists('data/ratings.csv'), \
        'File not exists in path, run preprocess.py before this.'
    print('Start..')
    start = time.time()
    if not os.path.exists('data/prank.graph'):
        Graph.gen_graph()
    if not os.path.exists('data/prank_1.model'):
        PersonalRank().train(user_id=1)
    movies = PersonalRank().predict(user_id=1)
    for movie in movies:
        print(movie)
    print('Cost time: %f' % (time.time() - start))
