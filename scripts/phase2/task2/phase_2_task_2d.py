import numpy as np
from scripts.phase2.common.task_1 import ActorTag
from scripts.phase2.common.util import Util


MAX_RATING = 5


class TagMovieRatingTensor(ActorTag):
    def __init__(self):
        super().__init__()

    def fetchTagMovieRatingTensor(self):
        mltags_df = self.data_extractor.get_mltags_data()
        mlratings_df = self.data_extractor.get_mlratings_data()

        movie_ratings_tags_df = mltags_df.merge(mlratings_df, how="left", on="movieid")

        tag_id_list = movie_ratings_tags_df["tagid"]
        tag_id_count = 0
        tag_id_dict = {}
        for element in tag_id_list:
            if element in tag_id_dict.keys():
                continue
            tag_id_dict[element] = tag_id_count
            tag_id_count += 1

        movieid_list = movie_ratings_tags_df["movieid"]
        movieid_count = 0
        movieid_dict = {}
        for element in movieid_list:
            if element in movieid_dict.keys():
                continue
            movieid_dict[element] = movieid_count
            movieid_count += 1

        tensor = np.zeros((tag_id_count + 1, movieid_count + 1, MAX_RATING + 1))

        util = Util()

        for index, row in movie_ratings_tags_df.iterrows():
            tagid = row["tagid"]
            movieid = row["movieid"]
            rating = row["rating"]
            if util.get_average_ratings_for_movie(movieid) <= rating:
                tagid_id = tag_id_dict[tagid]
                movieid_id = movieid_dict[movieid]
                tensor[tagid_id][movieid_id][rating] = 1

        return tensor

if __name__== "__main__":
    obj = TagMovieRatingTensor()
    tensor = obj.fetchTagMovieRatingTensor()
    util = Util()
    factors = util.CPDecomposition(tensor, 5)
    print(factors)