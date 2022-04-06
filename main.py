from typing import Any, Optional, List, Dict
from xmlrpc.client import boolean
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
import json
from recommender import get_recommend_items_by_knn, get_similar_items_by_knn, get_recommend_items_by_svd, get_similar_items_by_svd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
# original data
movie_info = pd.read_csv("movie_info.csv")

"""
=================== Body =============================
"""


class Movie(BaseModel):
    movie_id: int
    movie_title: str
    release_date: str
    score: int


class Recommend(BaseModel):
    is_svd: boolean
    movies: List[Movie]


# == == == == == == == == == API == == == == == == == == == == =

# show four genres
'''
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children"]}
'''

# show all generes


@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}


@app.post("/api/movies")
def get_movies(genre: list):
    print(genre)
    query_str = " or ".join(map(lambda x: ''+x+'==1', genre))
    results = movie_info.query(query_str)
    # results.loc[:, 'score'] = None
    res_temp = pd.DataFrame(results)
    res_temp.loc[:, 'score'] = None
    results = results.sample(
        18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend")
def get_recommend(recommend: Recommend):
    is_svd = recommend.is_svd
    movies = recommend.movies
    # print(movies)
    # iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    # score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    new_user_ratings = [(str(i.movie_id), int(i.score)) for i in movies]
    if is_svd:
        # print('Using SVD gets recommended items:')
        uid, based_iid_list, rec_list = get_recommend_items_by_svd(
            new_user_ratings)
    else:
        # print('Using KNN gets recommended items:')
        uid, based_iid_list, rec_list = get_recommend_items_by_knn(new_user_ratings)
    rec_iid_list = [int(i) for i, _ in rec_list]
    if len(rec_iid_list) > 12:
        rec_iid_list = rec_iid_list[:12]
    # print(rec_iid_list)
    based_movies = list(
        movie_info.loc[movie_info['movie_id'].isin(based_iid_list)]['movie_title'])
    rec_movies = movie_info.loc[movie_info['movie_id'].isin(rec_iid_list)]
    # print(rec_movies)
    # rec_movies.loc[:, 'like'] = None
    rec_temp = pd.DataFrame(rec_movies)
    rec_temp.loc[:, 'predicted'] = None
    rec_temp.loc[:, 'ground_truth'] = None
    rec_temp.loc[:, 'like'] = None
    for iid, pred in rec_list:
        rec_movies.loc[rec_movies['movie_id'] == int(iid), 'predicted'] = pred
    movies = rec_movies.loc[:, [
        'movie_id', 'movie_title', 'release_date', 'poster_url', 'predicted', 'ground_truth', 'like']]
    results = {'user_id': uid, 'based_movies': based_movies,
               'movies': movies.to_json(orient="records")}
    return json.loads(json.dumps(results))


@app.get("/api/add_recommend/{item_id}")
async def add_recommend(item_id: int, is_svd: boolean):
    if is_svd:
        # print('Using SVD gets similar items:')
        res = get_similar_items_by_svd(str(item_id), n=5)
    else:
        # print('Using KNN gets similar items:')
        res = get_similar_items_by_knn(str(item_id), n=5)
    res = [int(i) for i in res]
    # print(res)
    similar_movies = movie_info.loc[movie_info['movie_id'].isin(res)]
    # print(rec_movies)
    # rec_movies.loc[:, 'like'] = None
    rec_temp = pd.DataFrame(similar_movies)
    rec_temp.loc[:, 'like'] = None
    results = similar_movies.loc[:, [
        'movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
