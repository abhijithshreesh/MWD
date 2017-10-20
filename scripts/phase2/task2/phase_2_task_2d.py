import numpy as np

from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from scripts.phase2.common.util import Util


MAX_RATING = 5


class TagMovieRatingTensor():

    ordered_ratings = [0,1,2,3,4,5]
    ordered_movie_names = []
    ordered_tag_names = []

    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.tensor = self.fetchTagMovieRatingTensor()
        self.util = Util()
        self.factors = self.util.CPDecomposition(self.tensor, 5)

    def fetchTagMovieRatingTensor(self):
        mltags_df = self.data_extractor.get_mltags_data()

        util = Util()

        tag_id_list = mltags_df["tagid"]
        tag_id_count = 0
        tag_id_dict = {}
        for element in tag_id_list:
            if element in tag_id_dict.keys():
                continue
            tag_id_dict[element] = tag_id_count
            tag_id_count += 1
            name = util.get_tag_name_for_id(element)
            self.ordered_tag_names.append(name)

        movieid_list = mltags_df["movieid"]
        movieid_count = 0
        movieid_dict = {}
        for element in movieid_list:
            if element in movieid_dict.keys():
                continue
            movieid_dict[element] = movieid_count
            movieid_count += 1
            name = util.get_movie_name_for_id(element)
            self.ordered_movie_names.append(name)

        tensor = np.zeros((tag_id_count + 1, movieid_count + 1, MAX_RATING + 1))

        util = Util()

        for index, row in mltags_df.iterrows():
            tagid = row["tagid"]
            movieid = row["movieid"]
            avg_movie_rating = util.get_average_ratings_for_movie(movieid)
            for rating in range(0, int(avg_movie_rating) + 1):
                tagid_id = tag_id_dict[tagid]
                movieid_id = movieid_dict[movieid]
                tensor[tagid_id][movieid_id][rating] = 1

        return tensor

    def print_latent_semantics(self, r):
        i = 0
        for factor in self.factors:
            latent_semantics = self.util.get_latent_semantics(r, factor.transpose())
            self.util.print_latent_semantics(latent_semantics, self.get_factor_names(i))
            i+=1

    def get_factor_names(self, i):
        if i == 0:
            print("\n\nFor Tags:\n")
            return self.ordered_tag_names
        elif i == 1:
            print("\n\nFor Movies:\n")
            return self.ordered_movie_names
        elif i == 2:
            print("\n\nFor Ratings:\n")
            return self.ordered_ratings

if __name__== "__main__":
    obj = TagMovieRatingTensor()
    obj.print_latent_semantics(5)