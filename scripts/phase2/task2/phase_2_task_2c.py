import numpy as np

from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from scripts.phase2.common.util import Util


class ActorMovieYearTensor(object):

    ordered_years = []
    ordered_movie_names = []
    ordered_actor_names = []

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
            self.ordered_years.append(element)

        util = Util()
        movieid_list = movie_actor_df["movieid"]
        movieid_count = 0
        movieid_dict = {}
        for element in movieid_list:
            if element in movieid_dict.keys():
                continue
            movieid_dict[element] = movieid_count
            movieid_count += 1
            name = util.get_movie_name_for_id(element)
            self.ordered_movie_names.append(name)

        actorid_list = movie_actor_df["actorid"]
        actorid_count = 0
        actorid_dict = {}
        for element in actorid_list:
            if element in actorid_dict.keys():
                continue
            actorid_dict[element] = actorid_count
            actorid_count += 1
            name = util.get_actor_name_for_id(element)
            self.ordered_actor_names.append(name)

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
        i = 0
        for factor in self.factors:
            latent_semantics = self.util.get_latent_semantics(r, factor.transpose())
            self.util.print_latent_semantics(latent_semantics, self.get_factor_names(i))
            i+=1

    def get_factor_names(self, i):
        if i == 0:
            print("\n\nFor Years:\n")
            return self.ordered_years
        elif i == 1:
            print("\n\nFor Movies:\n")
            return self.ordered_movie_names
        elif i == 2:
            print("\n\nFor Actors:\n")
            return self.ordered_actor_names


if __name__ == "__main__":
    obj = ActorMovieYearTensor()
    obj.print_latent_semantics(5)
    # obj.print_partitioned_entites(5)
