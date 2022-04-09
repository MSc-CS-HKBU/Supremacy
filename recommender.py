import csv
import os
import random
from tkinter import N
import numpy as np
import pandas as pd
from scipy import spatial
from surprise import KNNWithMeans, SVD, Reader, Dataset, dump


def get_recommend_items_by_knn(new_user_ratings, n=12):
    random_rec = []
    based_iid_list, trainset = user_add(new_user_ratings)
    uid = trainset.n_users
    item_rid_list = [trainset.to_raw_iid(inner_id)
                     for inner_id in trainset.ir.keys()]
    print(f'number of total users: {trainset.n_users}')

    # algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo = KNNWithMeans(k=60, sim_options={
                        'name': 'cosine', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./rec.model', algo=algo, verbose=1)
    all_results = {}
    for i in item_rid_list:
        uid = str(uid)
        iid = str(i)
        pred = algo.predict(uid, iid).est
        if pred >= 5:
            all_results[iid] = pred
    for key in all_results:
        random_rec = random.sample(all_results.items(), 12)
    for i in random_rec:
        print(i)
    return uid, based_iid_list, random_rec


def get_recommend_items_by_svd(new_user_ratings, n=12):
    # rec_iid_list = []
    based_iid_list, trainset = user_add(new_user_ratings)
    uid = trainset.n_users
    item_rid_list = [trainset.to_raw_iid(inner_id)
                     for inner_id in trainset.ir.keys()]
    print(f'number of total users: {trainset.n_users}')
    # optimal parameters from Jupyter Notebook
    svd_algo = SVD(n_factors=20, n_epochs=30, biased=False)
    svd_algo.fit(trainset)
    dump.dump('./rec.model', algo=svd_algo, verbose=1)
    knn_algo = KNNWithMeans(k=60, sim_options={     # optimal parameters from Jupyter Notebook
        'name': 'cosine',
        'user_based': False
    })
    knn_algo.fit(trainset)
    dump.dump('./sim.model', algo=knn_algo, verbose=1)
    all_results = {}
    for i in item_rid_list:
        uid = str(uid)
        iid = str(i)
        pred = round(svd_algo.predict(uid, iid).est, 4)
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(),
                         key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in sorted_list[:n]:
        print(i)
    return uid, based_iid_list, sorted_list[:n]


def get_similar_items_by_knn(iid, n=12):
    algo = dump.load('./rec.model')[1]
    uid = str(algo.trainset.n_users)
    inner_id = algo.trainset.to_inner_iid(iid)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print("neighbors_iid", neighbors_iid)
    similar_items = {}
    for i in neighbors_iid:
        pred = round(algo.predict(uid, iid).est, 4)
        similar_items[i] = pred
    return similar_items


def get_similar_items_by_svd(iid, n=12):
    svd_algo = dump.load('./rec.model')[1]
    trainset = svd_algo.trainset
    uid = str(trainset.n_users)
    # print(f'iis: {iid}, number of total items: {algo.trainset.n_items}')
    # Option 1: calculating k nearest neighbors by KNNWithMeans algorithm, we use it here.
    knn_algo = dump.load('./sim.model')[1]
    inner_id = knn_algo.trainset.to_inner_iid(iid)
    neighbors = knn_algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [knn_algo.trainset.to_raw_iid(x) for x in neighbors]
    # Option 2: calculating k nearest neighbors by k_neighbors_item_based function, we don't use it because it executes too slowly.
    # neighbors_iid = k_neighbors_item_based(trainset, iid, n)
    print("neighbors_iid", neighbors_iid)
    similar_items = {}
    for i in neighbors_iid:
        pred = round(svd_algo.predict(uid, iid).est, 4)
        similar_items[i] = pred
    return similar_items


def user_add(new_user_ratings, is_reset=False):
    # simulate adding a new user into the original data file
    if is_reset or not os.path.exists('new_ratings.csv'):
        df = pd.read_csv('./ratings.csv')
        df.to_csv('new_' + 'ratings.csv', index=False)
    new_df = pd.read_csv('./new_ratings.csv', sep='\t', header=None,
                         names=['user', 'item', 'rating', 'timestamp'])
    uid = int(new_df['user'].max()) + 1
    scores = []
    with open(r'new_ratings.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        for (i, s) in new_user_ratings:
            if s > 0:
                new_rating = [str(uid), str(i), int(s), '0']
                scores.append([str(i), int(s)])
                wf.writerow(new_rating)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 2:
        sorted_scores = sorted_scores[:2]
    based_list = [int(s[0]) for s in sorted_scores]
    file_path = os.path.expanduser('new_ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    return based_list, trainset


# Calculating k nearest neighbors step by step
def k_neighbors_item_based(train_set, raw_iid, k):

    train_user_list = [u_id for u_id in train_set.ur.keys()]
    train_item_list = [i_id for i_id in train_set.ir.keys()]

    # initiate the matrix for storing the similarity values
    sim_item = np.empty((len(train_item_list), len(train_item_list)))

    # need to calculate the mean rating of each user for adjusted cosine similarity
    u_rating_mean_dict = {}
    for u_id in range(len(train_user_list)):
        sum_r = 0
        count = 0
        for (i, r) in train_set.ur[u_id]:
            sum_r += r
            count += 1
        u_rating_mean_dict[u_id] = sum_r / count

    # loop through all item to calculate the similaity
    for item1_id in range(len(train_item_list)):
        for item2_id in range(item1_id + 1, len(train_item_list)):
            # A: find common user that rated both i1 and i2
            item1_user_rating_dict = {}
            for (u, r) in train_set.ir[item1_id]:
                item1_user_rating_dict[u] = r
            item2_user_rating_dict = {}
            for (u, r) in train_set.ir[item2_id]:
                item2_user_rating_dict[u] = r
            common_users = list(set.intersection(set(item1_user_rating_dict.keys()),
                                                 set(item2_user_rating_dict.keys())))
            # B: construct rating vectors of common user for two items
            item1_rating_list, item2_rating_list = [], []
            for u in common_users:
                item1_rating_list.append(
                    item1_user_rating_dict[u] - u_rating_mean_dict[u])
                item2_rating_list.append(
                    item2_user_rating_dict[u] - u_rating_mean_dict[u])
            # C: calculate the similarities between two items, we use adjusted cosine similarity
            if len(common_users) > 0:
                sim = 1 - \
                    spatial.distance.cosine(
                        item1_rating_list, item2_rating_list)
            else:
                sim = 0
            # D: store measured similarity into matrix
            sim_item[item1_id][item2_id] = sim
            sim_item[item2_id][item1_id] = sim

    # calculating k nearset neighbors
    i_inner_id = train_set.to_inner_iid(raw_iid)

    neighbors_sim_dict = {}
    i_sim_item = sim_item[i_inner_id]
    for i in range(len(i_sim_item)):
        if i_inner_id != i:
            neighbors_sim_dict[i] = i_sim_item[i]

    k_neighbors = sorted(neighbors_sim_dict.items(),
                         reverse=True, key=lambda item: item[1])[:k]
    k_neighbors_raw_iid = [train_set.to_raw_iid(i) for i, _ in k_neighbors]

    return k_neighbors_raw_iid
