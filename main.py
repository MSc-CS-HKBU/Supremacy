from tabnanny import verbose
from textwrap import indent
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
import uvicorn
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset
import random
from surprise import accuracy
from recommender import get_recommend_items_by_svd, get_similar_items_by_svd


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
data = pd.read_csv("movies_update.csv")


"""
=================== Body =============================
"""


class Movie(BaseModel):
    movie_id: int
    movie_title: str
    release_date: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =

# show four genres
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children"]}

# show all generes
'''
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}
'''
# == == == == == == == == == KNN with Means == == == == == == == == == == =
@app.post("/api/movies")
def get_movies(genre: list):
    print(genre)
    query_str = " or ".join(map(map_genre, genre))
    results = data.query(query_str)
    results.loc[:, 'score'] = None
    results = results.sample(18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend_knn")
def get_recommend(movies: List[Movie]):
    # print(movies)
    iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    new_user_ratings = [(str(i.movie_id), int(i.score)) for i in movies]
    uid, res = get_initial_items(new_user_ratings)
    # res是列表
    res_id = []
    for i in res:
        res_id.append(int(i[0]))
    # res = get_initial_items(iid,score)
    if len(res_id) > 12:
        res_id = res_id[:12]
    rec_movies = data.loc[data['movie_id'].isin(res_id)]
    rec_temp = pd.DataFrame(rec_movies)
    rec_temp.loc[:, 'like'] = None
    rec_temp.loc[:, 'ground_truth'] = None
    rec_temp.loc[:, 'pred'] = None
    for i in range(12):
        rec_movies.iloc[i, 27] = res[i][1]
    movies = rec_movies.loc[:, [
        'movie_id', 'movie_title', 'release_date', 'poster_url', 'like', 'ground_truth', 'pred']]
    results = {'user_id': uid, 'movies': movies.to_json(orient="records")}
    return json.loads(json.dumps(results))


@app.get("/api/add_recommend/{item_id}&{uid}")
async def add_recommend(item_id, uid):
    res = get_similar_items_knn(uid, str(item_id), n=5)
    # res是字典
    res_id = []
    for key in res:
        res_id.append(int(key))
    # res = [int(i) for i in res]
    similar_movies = data.loc[data['movie_id'].isin(res_id)]
    rec_temp = pd.DataFrame(similar_movies)
    rec_temp.loc[:, 'like'] = None
    rec_temp.loc[:, 'ground_truth'] = None
    rec_temp.loc[:, 'pred'] = None
    for i, key in zip(range(5), res):
        similar_movies.iloc[i, 27] = res[key]
    results = similar_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like','ground_truth','pred']]
    return json.loads(results.to_json(orient="records"))

# evaluation & accuracy
@app.post("/api/evaluation")
async def run_eval(movies:list):
    predictions = []
    ratings = []
    for i in movies:
        predictions.append([i['user_id'], i['movie_id'], i['ground_truth'], i['pred'], False])
    for i in range(len(predictions)):
        print(predictions[i][2])
        ratings.append(predictions[i][2])
    avg_rating = np.mean(ratings)
    print('avg', avg_rating)
    accuracy.rmse(predictions, verbose=True)


def user_add(new_user_ratings, is_reset=False):
    # simulate adding a new user into the original data file
    if is_reset or not os.path.exists('new_ratings.csv'):
        df = pd.read_csv('./ratings.csv')
        df.to_csv('new_' + 'ratings.csv', index=False)
    new_df = pd.read_csv('./new_ratings.csv', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'],index_col=None)
    uid = int(new_df['user'].max()) + 1
    with open(r'new_ratings.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        for (i, s) in new_user_ratings:
            if s > 0:
                new_rating = [str(uid), str(i), int(s), '0']
                wf.writerow(new_rating)
    file_path = os.path.expanduser('new_ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    return trainset

def get_initial_items(new_user_ratings, n=12):
    random_rec = []
    trainset = user_add(new_user_ratings)
    uid = trainset.n_users
    item_rid_list = [trainset.to_raw_iid(inner_id) for inner_id in trainset.ir.keys()]
    print(f'number of total users: {trainset.n_users}')

    # algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo = KNNWithMeans(k=60, sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in item_rid_list:
        uid = str(uid)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        if pred >= 5:
            all_results[iid] = pred
    for key in all_results:
        random_rec = random.sample(all_results.items(), n)

    return uid, random_rec


def get_similar_items_knn(uid, iid, n):
    algo = dump.load('./model')[1]
    print("iid", iid)
    inner_id = algo.trainset.to_inner_iid(iid)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    rec_similar = {}
    for i in neighbors_iid:
        pred = algo.predict(uid, iid).est
        rec_similar[i] = pred
    print("neighbors_iid", neighbors_iid)
    return rec_similar



    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)