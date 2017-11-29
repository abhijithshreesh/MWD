import pandas as pd
import os
import random
import math
from scipy.spatial import distance
import numpy
import json
from config_parser import ParseConfig
from phase_3.scripts.util import Util

conf = ParseConfig()

class MovieLSH():
    def __init__(self, num_layers, num_hashs):
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


    def assign_group(self, value):
        if value < 0:
            return math.floor(value/self.w_length)
        else:
            return math.ceil(value / self.w_length)


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


    def project_on_hash_function(self, movie_vector, lsh_vector):
        movie_lsh_dot_product = numpy.dot(movie_vector, lsh_vector)
        if movie_lsh_dot_product == 0.0:
            return 0
        lsh_vector_dot_product = numpy.dot(lsh_vector, lsh_vector)
        projection = movie_lsh_dot_product/lsh_vector_dot_product*lsh_vector
        projection_magnitude = numpy.linalg.norm(projection)
        return projection_magnitude


    def LSH(self, vector):
        bucket_list = []
        for lsh_vector in range(0, len(self.lsh_points_dict)):
            bucket_list.append(self.assign_group(self.project_on_hash_function(numpy.array(vector), numpy.array(self.lsh_points_dict[lsh_vector]))))
        return bucket_list


    def group_data(self):
        U_dataframe = pd.DataFrame(self.U)
        U_dataframe = U_dataframe[U_dataframe.columns[0:500]]
        self.init_lsh_vectors(U_dataframe)
        self.w_length = min(self.lsh_range_dict.values()) / 10
        self.column_groups = {vector: [] for vector in self.lsh_range_dict.keys()}
        bucket_matrix = numpy.zeros(shape=(len(self.U), len(self.lsh_points_dict)))
        self.U_matrix = U_dataframe.values

        for movie in range(0, len(self.U_matrix)):
            bucket_matrix[movie] = self.LSH(self.U_matrix[movie])

        movie_df = self.movie_tag_df.reset_index()
        movie_id_df = pd.DataFrame(movie_df["movieid"])
        self.movie_latent_df = U_dataframe.join(movie_id_df, how="left")
        self.movie_latent_df.to_csv(os.path.join(self.data_set_loc, "movie_latent_semantic.csv"), index=False)
        return pd.DataFrame(bucket_matrix).join(movie_id_df, how="left")


    def index_data(self, df):
        index_structure_dict = {}
        for index, row in df.iterrows():
            movie_id = row["movieid"]
            column = 0
            for i in range(0, self.num_layers):
                bucket = ""
                for j in range(0, self.num_hashs):
                    interval = row[column]
                    bucket = bucket + str(int(interval)) + "."
                    column += 1
                    if bucket.strip(".") in index_structure_dict:
                        index_structure_dict[bucket.strip(".")].add(movie_id)
                    else:
                        movie_set = set()
                        movie_set.add(movie_id)
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
        self.movie_bucket_df = self.group_data()
        movie_list_bucket_df = self.movie_bucket_df[self.movie_bucket_df["movieid"].isin(movie_list)] if movie_list  else self.movie_bucket_df
        self.index_structure = self.index_data(movie_list_bucket_df)


    def query_for_nearest_neighbours_for_movie(self, query_movie_id, no_of_nearest_neighbours):
        query_movie_name = self.util.get_movie_name_for_id(query_movie_id)
        print("\nQuery Movie Name : " + query_movie_name + " - " + str(int(query_movie_id)) + "\n")
        query_vector = self.movie_latent_df[self.movie_latent_df["movieid"] == query_movie_id]
        query_vector = query_vector.iloc[0].tolist()[0:-1]
        return self.query_for_nearest_neighbours(query_vector, no_of_nearest_neighbours)


    def query_for_nearest_neighbours(self, query_vector, no_of_nearest_neighbours):
        query_bucket_list = self.LSH(query_vector)
        query_hash_key_list = self.fetch_hash_keys(query_bucket_list)
        selected_movie_set = set()
        nearest_neighbour_list = {}
        for j in range(0, self.num_hashs):
            for i in range(0, len(query_hash_key_list)):
                selected_movie_set.update(self.index_structure.get(query_hash_key_list[i].rsplit(".", j)[0], ''))
                selected_movie_set.discard('')
                selected_movie_vectors = self.movie_latent_df[self.movie_latent_df["movieid"].isin(selected_movie_set)]
                distance_from_query_list = []
                for i in range(0, len(selected_movie_vectors.index)):
                    row_list = selected_movie_vectors.iloc[i].tolist()
                    euclidean_distance = distance.euclidean(row_list[0:-1], query_vector)
                    if(euclidean_distance != 0):
                        distance_from_query_list.append((row_list[-1], euclidean_distance))
                distance_from_query_list = sorted(distance_from_query_list, key=lambda x: x[1])
                nearest_neighbour_list = ([each[0] for each in distance_from_query_list[0:no_of_nearest_neighbours]])
                if (len(nearest_neighbour_list) >= no_of_nearest_neighbours):
                    break
        nearest_neighbours = [int(each) for each in nearest_neighbour_list]
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
    movie_list = []
    while True:
        query_movie = int(input("\nEnter Query Movie ID : "))
        no_of_nearest_neighbours = int(input("\nEnter No. of Nearest Neighbours : "))

        movie_lsh = MovieLSH(num_layers, num_hashs)
        with open(os.path.join(movie_lsh.data_set_loc, 'task_3_details.txt'), 'w') as outfile:
            outfile.write(json.dumps({"num_layers": num_layers,
                        'num_hashs': num_hashs,
                        "movie_list": movie_list,
                        "query_movie":query_movie,
                        "no_of_nearest_neighbours" : no_of_nearest_neighbours},
                       sort_keys=True, indent = 4, separators = (',', ': ')))
        movie_lsh.create_index_structure(movie_list)
        nearest_neighbours = movie_lsh.query_for_nearest_neighbours_for_movie(query_movie, no_of_nearest_neighbours)
        movie_lsh.util.print_movie_recommendations_and_collect_feedback(nearest_neighbours, 3, None)

        confirmation = input("\n\nDo you want to continue? (y/Y/n/N): ")
        if confirmation != "y" and confirmation != "Y":
            break
