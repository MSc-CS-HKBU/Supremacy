import os
import numpy as np
from surprise import accuracy


def get_evaluate(is_svd, movies):
    predictions = [[str(movie.user_id), str(movie.movie_id),
                    movie.ground_truth, movie.predict, False] for movie in movies]
    ratings = [movie.ground_truth for movie in movies]
    avg_rating = np.mean(ratings)
    # print(f'average rating: {avg_rating}')
    rmse = accuracy.rmse(predictions, verbose=True)
    # print(f'user_rmse: {rmse}')
    if is_svd:
        with open('svd_rmse.csv', mode='a', newline='', encoding='utf8') as cfa:
            cfa.write(str(rmse) + '\n')
    else:
        with open('knn_rmse.csv', mode='a', newline='', encoding='utf8') as cfa:
            cfa.write(str(rmse) + '\n')
    if not os.path.exists('knn_rmse.csv'):
        knn_rmse = []
    else:
        with open('knn_rmse.csv', 'r') as fr:
            knn_lines = fr.readlines()
        knn_rmse = [float(knn.rstrip('\n')) for knn in knn_lines]
    if not os.path.exists('svd_rmse.csv'):
        svd_rmse = []
    else:
        with open('svd_rmse.csv', 'r') as fr:
            svd_lines = fr.readlines()
        svd_rmse = [float(svd.rstrip('\n')) for svd in svd_lines]
    return knn_rmse, svd_rmse, avg_rating

def rm_file(knn_rmse, svd_rmse):
    if len(knn_rmse) >= 2:
        os.remove('knn_rmse.csv')
    if len(svd_rmse) >= 2:
        os.remove('svd_rmse.csv')
