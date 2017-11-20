import numpy as np

from config_parser import ParseConfig
from data_extractor import DataExtractor
from util import Util
from phase1_task_2 import GenreTag


class MovieTagGenreTensor(object):

    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.genre_tag_data = GenreTag()
        self.ordered_genre = []
        self.ordered_movie_names = []
        self.ordered_tag = []
        self.print_list = ["\n\nFor Movies:", "\n\nFor Genres:", "\n\nFor Tags:"]
        self.util = Util()
        self.tensor = self.fetchMovieGenreTagTensor()
        self.factors = self.util.CPDecomposition(self.tensor, 10)

    def fetchMovieGenreTagTensor(self):
        """
        Create actor movie year tensor
        :return: tensor
        """
        #movies_df = self.data_extractor.get_mlmovies_data()
        #actor_df = self.data_extractor.get_movie_actor_data()
        genre_data = self.genre_tag_data.get_genre_data()
        #genre_data.to_csv('genre_data.csv', index=True, encoding='utf-8')
        #movie_actor_df = actor_df.merge(movies_df, how="left", on="movieid")
        movie_list = genre_data["moviename"]
        movie_count = 0
        movie_dict = {}
        for element in movie_list:
            if element in movie_dict.keys():
                continue
            movie_dict[element] = movie_count
            movie_count += 1
            self.ordered_movie_names.append(element)

        genre_list = genre_data["genre"]
        genre_count = 0
        genre_dict = {}
        for element in genre_list:
            if element in genre_dict.keys():
                continue
            genre_dict[element] = genre_count
            genre_count += 1
            self.ordered_genre.append(element)

        tag_list = genre_data["tag"]
        tag_count = 0
        tag_dict = {}
        for element in tag_list:
            if element in tag_dict.keys():
                continue
            tag_dict[element] = tag_count
            tag_count += 1
            self.ordered_tag.append(element)

        tensor = np.zeros((movie_count, genre_count, tag_count))

        for index, row in genre_data.iterrows():
            movie = row["moviename"]
            genre = row["genre"]
            tag = row["tag"]
            movie_name = movie_dict[movie]
            genre_name = genre_dict[genre]
            tag_name = tag_dict[tag]
            tensor[movie_name][genre_name][tag_name] = 1

        return tensor

    def print_latent_semantics(self, r):
        """
        Pretty print latent semantics
        :param r:
        """
        i = 0
        for factor in self.factors:
            print(self.print_list[i])
            latent_semantics = self.util.get_latent_semantics(r, factor.transpose())
            self.util.print_latent_semantics(latent_semantics, self.get_factor_names(i))
            i += 1

    def get_factor_names(self, i):
        """
        Obtain factor names
        :param i:
        :return: factor names
        """
        if i == 0:
            return self.ordered_movie_names
        elif i == 1:
            return self.ordered_genre
        elif i == 2:
            return self.ordered_tag

    def get_partitions(self, no_of_partitions):
        """
        Partition factor matrices
        :param no_of_partitions:
        :return: list of groupings
        """
        i = 0
        groupings_list = []
        for factor in self.factors:
            groupings = self.util.partition_factor_matrix(factor, no_of_partitions, self.get_factor_names(i))
            groupings_list.append(groupings)
            i += 1

        return groupings_list

    def print_partitioned_entities(self, no_of_partitions):
        """
        Pretty print groupings
        :param no_of_partitions:
        """
        groupings_list = self.get_partitions(no_of_partitions)
        i = 0
        for groupings in groupings_list:
            print(self.print_list[i])
            self.util.print_partitioned_entities(groupings)
            i += 1


if __name__ == "__main__":
    obj = MovieTagGenreTensor()
    obj.print_latent_semantics(10)
    obj.print_partitioned_entities(5)
