import numpy as np

from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from scripts.phase2.common.util import Util


class ActorMovieYearTensor(object):
    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.tensor = self.fetchActorMovieYearTensor()
        self.util = Util()
        self.factors = self.util.CPDecomposition(self.tensor, 5)

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

    def print_latent_semantics(self, r):
        for factor in self.factors:
            latent_semantics = self.util.get_latent_semantics(r, factor.transpose())
            self.util.print_latent_semantics(latent_semantics, )


if __name__ == "__main__":
    obj = ActorMovieYearTensor()
    obj.print_latent_semantics(5)
    # obj.print_partitioned_entites(5)
