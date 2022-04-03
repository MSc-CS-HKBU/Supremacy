import csv
import os
import numpy as np
import pandas as pd
from scipy import spatial
from surprise import SVD, Reader, Dataset, dump
from surprise.model_selection import GridSearchCV


n_factors_list = [10, 20, 30]
n_epochs_list = [10, 20]

def get_recommend_items_by_svd(new_user_ratings, n=12, uid=611):
    rec_iid_list = []
    user_add(uid, new_user_ratings)
    file_path = os.path.expanduser('new_ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = best_algo_by_svd(data)
    algo.fit(trainset)
    dump.dump('./model', algo=algo, verbose=1)
    all_results = {}
    for i in range(trainset.n_items):
        uid = str(uid)
        iid = str(i)
        pred = algo.predict(uid, iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(),
                         key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        rec_iid_list.append(sorted_list[i][0])
    return rec_iid_list


def get_similar_items_by_svd(iid, n=12):
    algo = dump.load('./model')[1]
    # print(f'iis: {iid}, number of total items: {algo.trainset.n_items}')
    # inner_id = algo.trainset.to_inner_iid(iid)
    # neighbors = algo.get_neighbors(inner_id, k=n)
    # neighbors_iid_list = [algo.trainset.to_raw_iid(x) for x in neighbors]
    neighbors_iid_list = k_neighbors_item_based(algo.trainset, iid, n)
    print("neighbors_iid", neighbors_iid_list)
    return neighbors_iid_list

def user_add(uid, new_user_ratings):
    # simulate adding a new user into the original data file
    df = pd.read_csv('./ratings.csv')
    df.to_csv('new_' + 'ratings.csv', index=False)
    with open(r'new_ratings.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        for (i, s) in new_user_ratings:
            if s > 0:
                new_rating = [str(uid), str(i), int(s), '0']
                wf.writerow(new_rating)

def best_algo_by_svd(data):
    svd_param_grid = {'n_factors': n_factors_list,
                      'n_epochs': n_epochs_list, 'biased': [True]}
    svd_gs = GridSearchCV(SVD, svd_param_grid, measures=['rmse'], cv=3)
    svd_gs.fit(data)
    svd_best_algo = svd_gs.best_estimator['rmse']
    return svd_best_algo


# Calculating simmilarity by Cosine Similarity
def k_neighbors_item_based(train_set, raw_iid, k):
    
    train_user_list = [u_id  for u_id in train_set.ur.keys()]
    train_item_list = [i_id  for i_id in train_set.ir.keys()]

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
                item1_rating_list.append(item1_user_rating_dict[u] - u_rating_mean_dict[u])
                item2_rating_list.append(item2_user_rating_dict[u] - u_rating_mean_dict[u])
            # C: calculate the similarities between two items, we use adjusted cosine similarity
            if len(common_users) > 0:
                sim = 1 - spatial.distance.cosine(item1_rating_list,item2_rating_list)
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

    k_neighbors = sorted(neighbors_sim_dict.items(), reverse=True, key=lambda item: item[1])[:k]
    k_neighbors_raw_iid = [train_set.to_raw_iid(i) for i, _ in k_neighbors]

    return k_neighbors_raw_iid