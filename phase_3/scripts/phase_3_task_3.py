import pandas as pd
import random
from phase_3_task_1 import UserMovieRecommendation
from scipy.spatial import distance
import numpy

class MovieLSH(UserMovieRecommendation):

    def __init__(self, num_layers, num_hashs, movie_id_list):
        super().__init__()
        self.movie_tag_df = self.get_movie_tag_matrix()
        self.num_layers = num_layers
        self.num_hashs = num_hashs
        self.movie_id_list = movie_id_list
        self.latent_range_dict = {}
        self.lsh_points_dict = {}
        self.lsh_range_dict ={}
        (self.U, self.s, self.Vt) = self.util.SVD(self.movie_tag_df.values)

    def assign_group(self, value, group_range):
        for i in range(0,len(group_range)):
            if value>=group_range[i] and value<group_range[i+1]:
                return i

    def init_lsh_vectors(self, U_dataframe):
        origin = list(numpy.zeros(shape=(1, 500)))
        for column in U_dataframe:
            self.latent_range_dict[column] = (U_dataframe[column].min(), U_dataframe[column].max())
        for i in range(0, self.num_layers*self.num_hashs):
            cur_vector_list = []
            for column in U_dataframe:
                cur_vector_list.append(random.uniform(self.latent_range_dict[column][0], self.latent_range_dict[column][1]))
            self.lsh_points_dict[i] = cur_vector_list
            self.lsh_range_dict[i] = distance.euclidean(origin, cur_vector_list)

    def group_data(self):
        U_dataframe = pd.DataFrame(self.U)
        U_dataframe = U_dataframe[U_dataframe.columns[0:500]]
        self.init_lsh_vectors(U_dataframe)
        group_length = {}
        column_groups = {}

        for vector, vector_range in list(self.lsh_range_dict.items()):
            group_length[vector] = vector_range/10
            column_groups[vector] = []

        for vector, vector_range in list(self.lsh_range_dict.items()):
            interval_min = 0
            for i in range(0, 11):
                column_groups[vector].append(interval_min)
                interval_min = interval_min+group_length[vector]
            column_groups[vector][-1] += 1
        interval_matrix = numpy.zeros(shape=(len(self.U), len(self.lsh_points_dict)))
        U_matrix = U_dataframe.values
        for vector in range(0, len(self.lsh_points_dict)):
            for row in range(0, len(U_matrix)):
                interval_matrix[row][vector] = self.assign_group(numpy.dot(numpy.array(U_matrix[row]), numpy.array(self.lsh_points_dict[vector])), column_groups[vector])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_2_task_1a.py 146',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    num_layers = 3
    num_hashs = 4
    movie_id_list = []
    obj = MovieLSH(num_layers, num_hashs, movie_id_list)
    obj.group_data()