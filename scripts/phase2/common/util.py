import math

from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor

conf = ParseConfig()


class Util(object):

    """
    Class to relate actors and tags.
    """

    def __init__(self):
        """
        Initializing the data extractor object to get data from the csv files
        """
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.mlratings = self.data_extractor.get_mlratings_data()
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.imdb_actor_info = self.data_extractor.get_imdb_actor_info_data()

    def get_sorted_actor_ids(self):
        actor_info = self.data_extractor.get_imdb_actor_info_data()
        actorids = actor_info.id
        actorids = actorids.sort_values()
        return actorids

    def get_movie_id(self, movie):
        all_movie_data = self.mlmovies
        movie_data = all_movie_data[all_movie_data['moviename'] == movie]
        movie_id = movie_data['movieid'].unique()

        return movie_id[0]

    def get_average_ratings_for_movie(self, movie_id):
        all_ratings = self.mlratings
        movie_ratings = all_ratings[all_ratings['movieid'] == movie_id]

        ratings_sum = 0
        ratings_count = 0
        for index, row in movie_ratings.iterrows():
            ratings_count += 1
            ratings_sum += row['rating']

        return ratings_sum / float(ratings_count)

    def get_actor_name_for_id(self, actor_id):
        actor_data = self.imdb_actor_info[self.imdb_actor_info['id'] == actor_id]
        name = actor_data['name'].unique()

        return name[0]

    def partition_factor_matrix(self, matrix, no_of_partitions, entity_names):
        entity_dict = {}
        for i in range(0, len(matrix)):
            length = 0
            for latent_semantic in matrix[i]:
                length += abs(latent_semantic)
            entity_dict[entity_names[i]] = length

        max_length = max(entity_dict.values())
        min_length = min(entity_dict.values())
        length_of_group = (float(max_length) - float(min_length)) / float(no_of_partitions)

        groups = {}
        for key in entity_dict.keys():
            entity_length = entity_dict[key]
            group_no = math.ceil(float(entity_length - min_length) / float(length_of_group))
            if group_no == 0:
                group_no = 1
            if "Group " + str(group_no) in groups.keys():
                groups["Group " + str(group_no)].append(key)
            else:
                groups["Group " + str(group_no)] = []
                groups["Group " + str(group_no)].append(key)

        return groups


if __name__ == "__main__":
    obj = Util()
    actorids = obj.get_sorted_actor_ids()
    print("Actorids Sorted along with original index: \n")
    print(actorids)
