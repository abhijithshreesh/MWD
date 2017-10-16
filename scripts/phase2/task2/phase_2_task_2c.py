import tensorly.tensorly.decomposition as decomp
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_1 import ActorTag
import pandas as pd
import numpy as np
conf = ParseConfig()

class ActorMovieYearTensor(ActorTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def fetchActorMovieYearTensor(self):
        movies_df = self.data_extractor.get_mlmovies_data()
        actor_df = self.data_extractor.get_movie_actor_data()

        movie_actor_df = actor_df.merge(movies_df, how="left", on="movieid")
        year_list = movie_actor_df["year"]
        year_count = 0
        year_dict = {}
        for element in year_list:
            if element in year_dict.keys():
                continue
            year_dict[element] = year_count
            year_count += 1

        movieid_list = movie_actor_df["movieid"]
        movieid_count = 0
        movieid_dict = {}
        for element in movieid_list:
            if element in movieid_dict.keys():
                continue
            movieid_dict[element] = movieid_count
            movieid_count += 1

        actorid_list = movie_actor_df["actorid"]
        actorid_count = 0
        actorid_dict = {}
        for element in actorid_list:
            if element in actorid_dict.keys():
                continue
            actorid_dict[element] = actorid_count
            actorid_count += 1

        tensor = np.zeros((year_count + 1, movieid_count + 1, actorid_count + 1))

        for index, row in movie_actor_df.iterrows():
            year = row["year"]
            movieid = row["movieid"]
            actorid = row["actorid"]
            year_id = year_dict[year]
            movieid_id = movieid_dict[movieid]
            actorid_id = actorid_dict[actorid]
            tensor[year_id][movieid_id][actorid_id] = 1

        return tensor

    def cpDecomposition(self, tensor):
        factors = decomp.parafac(tensor, 5)
        return factors

if __name__ == "__main__":
    obj = ActorMovieYearTensor()
    tensor = obj.fetchActorMovieYearTensor()
    factors = obj.cpDecomposition(tensor)
    print(factors)