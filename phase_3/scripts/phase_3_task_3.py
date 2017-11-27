import pandas as pd
import os
import random
import math
from scipy.spatial import distance
import numpy
from config_parser import ParseConfig
from phase_3.scripts.util import Util

conf = ParseConfig()

class MovieLSH():
    def __init__(self, num_layers = 0, num_hashs = 0, fileName = "lsh_index_structure.csv"):
        self.util = Util()
        self.movie_tag_df = self.util.get_movie_tag_matrix()
        self.num_layers = num_layers
        self.num_hashs = num_hashs
        self.latent_range_dict = {}
        self.lsh_points_dict = {}
        self.lsh_range_dict = {}
        self.column_groups = []
        self.U_matrix = []
        self.movie_bucket_df = pd.DataFrame()
        self.movie_latent_df = pd.DataFrame()
        self.w_length = 0.0
        (self.U, self.s, self.Vt) = self.util.SVD(self.movie_tag_df.values)
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.fileName = fileName


    def assign_group(self, value):
        if value < 0:
            return math.floor(value/self.w_length)
        else:
            return math.ceil(value/self.w_length)

    def init_lsh_vectors(self, U_dataframe):
        origin = list(numpy.zeros(shape=(1, 500)))
        for column in U_dataframe:
            self.latent_range_dict[column] = (U_dataframe[column].min(), U_dataframe[column].max())
        for i in range(0, self.num_layers * self.num_hashs):
            cur_vector_list = []
            for column in U_dataframe:
                cur_vector_list.append(random.uniform(self.latent_range_dict[column][0], self.latent_range_dict[column][1]))
            self.lsh_points_dict[i] = cur_vector_list
            self.lsh_range_dict[i] = distance.euclidean(origin, cur_vector_list)


    def LSH(self, vector):
        bucket_list = []
        for lsh_vector in range(0, len(self.lsh_points_dict)):
            bucket_list.append(self.assign_group(numpy.dot(numpy.array(vector), numpy.array(self.lsh_points_dict[lsh_vector]))))
        return bucket_list

    def group_data(self):
        U_dataframe = pd.DataFrame(self.U)
        U_dataframe = U_dataframe[U_dataframe.columns[0:500]]
        self.init_lsh_vectors(U_dataframe)
        self.w_length = min(self.lsh_range_dict.values()) / 20
        self.column_groups = {vector: [] for vector in self.lsh_range_dict.keys()}
        bucket_matrix = numpy.zeros(shape=(len(self.U), len(self.lsh_points_dict)))
        self.U_matrix = U_dataframe.values

        for movie in range(0, len(self.U_matrix)):
            bucket_matrix[movie] = self.LSH(self.U_matrix[movie])

        movie_df = self.movie_tag_df.reset_index()
        movie_name_df = pd.DataFrame(movie_df["moviename"])
        self.movie_latent_df = U_dataframe.join(movie_name_df, how="left")
        self.movie_latent_df.to_csv(os.path.join(self.data_set_loc, "movie_latent_semantic.csv"), index=False)
        return pd.DataFrame(bucket_matrix).join(movie_name_df, how="left")

    def index_data(self, df):
        index_structure_dict = {}
        for index, row in df.iterrows():
            movie_name = row["moviename"]
            column = 0
            for i in range(0, self.num_layers):
                bucket = ""
                for j in range(0, self.num_hashs):
                    interval = row[column]
                    bucket = bucket + str(int(interval)) + "."
                    column += 1
                    if bucket.strip(".") in index_structure_dict:
                        index_structure_dict[bucket.strip(".")].add(movie_name)
                    else:
                        movie_set = set()
                        movie_set.add(movie_name)
                        index_structure_dict[bucket.strip(".")] = movie_set
        return index_structure_dict

    def fetch_hash_keys(self, bucket_list):
        column = 0
        hash_key_list = []
        for i in range(0, self.num_layers):
            bucket = ""
            for j in range(0, self.num_hashs):
                interval = bucket_list[column]
                if(j != self.num_hashs-1):
                    bucket = bucket + str(int(interval)) + "."
                else:
                    bucket = bucket + str(int(interval))
                column += 1
            hash_key_list.append(bucket)
        return hash_key_list

    def create_index_structure(self, movie_list):
        self.movie_bucket_df = obj.group_data()
        movie_name_list = [self.util.get_movie_name_for_id(movie_id) for movie_id in movie_list]
        movie_list_latent_df = self.movie_latent_df[self.movie_latent_df["moviename"].isin(movie_name_list)]
        movie_list_bucket_df = self.movie_bucket_df[self.movie_bucket_df["moviename"].isin(movie_name_list)]

        self.index_structure = obj.index_data(movie_list_bucket_df)
        temp_index_structure = dict([(k, pd.Series(list(v))) for k, v in list(self.index_structure.items())])
        pd.DataFrame.from_dict(temp_index_structure).to_csv(os.path.join(self.data_set_loc, self.fileName), index=False)

    def query_for_nearest_neighbours_for_movie(self, query_movie_id, no_of_nearest_neighbours):
        query_movie_name = self.util.get_movie_name_for_id(query_movie_id)
        query_vector = self.movie_latent_df[self.movie_latent_df["moviename"] == query_movie_name]
        query_vector = query_vector.iloc[0].tolist()[0:-1]
        return self.query_for_nearest_neighbours(query_vector, no_of_nearest_neighbours)

    def query_for_nearest_neighbours_using_csv(self, query_vector, no_of_nearest_neighbours):
        index_structure_df = pd.read_csv(os.path.join(self.data_set_loc, self.fileName)).fillna("")
        index_structure_df = index_structure_df.drop(['Unnamed: 0'], axis=1)
        self.index_structure = index_structure_df.to_dict("list")
        self.index_structure = {k: set(v) for k,v in list(self.index_structure.items())}
        for k,v in list(self.index_structure.items()):
            v.discard('')
        return self.query_for_nearest_neighbours(query_vector, no_of_nearest_neighbours)

    def query_for_nearest_neighbours(self, query_vector, no_of_nearest_neighbours):
        query_bucket_list = self.LSH(query_vector)
        query_hash_key_list = self.fetch_hash_keys(query_bucket_list)
        selected_movie_set = set()
        for j in range(0, self.num_hashs):
            for i in range(0, len(query_hash_key_list)):
                selected_movie_set.update(self.index_structure.get(query_hash_key_list[i].rsplit(".", j)[0], ''))
                selected_movie_set.discard('')
                selected_movie_vectors = self.movie_latent_df[self.movie_latent_df["moviename"].isin(selected_movie_set)]
                distance_from_query_list = []
                for i in range(0, len(selected_movie_vectors.index)):
                    row_list = selected_movie_vectors.iloc[i].tolist()
                    distance_from_query_list.append((row_list[-1], distance.euclidean(row_list[0:-1], query_vector)))
                distance_from_query_list = sorted(distance_from_query_list, key=lambda x: x[1])
                nearest_neighbours = [each[0] for each in distance_from_query_list[0:no_of_nearest_neighbours+1]]
                if (len(nearest_neighbours) == no_of_nearest_neighbours):
                    break
        return nearest_neighbours


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_2_task_1a.py 146',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    num_layers = 4
    num_hashs = 3
    movie_list = [7755, 584, 4609, 9336, 9942, 2190, 9975, 6426]
    query_movie = 9942
    no_of_nearest_neighbours = 5
    obj = MovieLSH(num_layers, num_hashs)
    obj.create_index_structure(movie_list)
    nearest_neighbours = obj.query_for_nearest_neighbours_for_movie(query_movie, no_of_nearest_neighbours)
    obj.util.print_movie_recommendations_and_collect_feedback(nearest_neighbours, 3, None)
