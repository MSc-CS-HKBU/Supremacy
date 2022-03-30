import pandas as pd

name = ["movie_id", "movie_title", "release_date", "IMDb URL", "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"]
data = []
with open("u.item", 'r') as f:
    lines = f.readlines()
    for line in lines:
        info = line.strip().split("||")
        l = info[0].split("|") + info[1].split("|")
        data.append(l)
# print(data)

right = pd.read_csv("movie_poster.csv")
# print(right)
ori_frame = pd.DataFrame(data, columns=name)
# print(ori_frame)
ori_frame[['movie_id']] = ori_frame[['movie_id']].astype(int)
rs = pd.merge(ori_frame, right, on='movie_id', how='left')
print(rs.shape)
rs.to_csv("movie_info.csv", index=None)
