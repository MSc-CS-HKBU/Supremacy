from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
import  json
from utils import map_genre
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
# original data
ori_data = pd.read_csv("movie_info.csv")

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


@app.post("/api/movies")
def get_movies(genre: list):
    print(genre)
    query_str = " or ".join(map(map_genre, genre))
    results = ori_data.query(query_str)
    results.loc[:, 'score'] = None
    results = results.sample(
        18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
    # print(movies)
    # iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    # score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    new_user_ratings = [(str(i.movie_id), int(i.score)) for i in movies]
    res = get_recommend_items_by_svd(new_user_ratings)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print(res)
    rec_movies = ori_data.loc[ori_data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, [
        'movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


@app.get("/api/add_recommend/{item_id}")
async def add_recommend(item_id):
    res = get_similar_items_by_svd(str(item_id), n=5)
    res = [int(i) for i in res]
    print(res)
    rec_movies = ori_data.loc[ori_data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)