import os
import config_parser
import data_extractor
import pandas as pd
from phase_3_task_3 import MovieLSH
from util import Util
import json


class NearestNeighbourBasedRelevanceFeedback(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.util = Util()
        self.movies_dict = {}
        self.movie_tag_matrix = self.get_movie_tag_matrix()
        task_3_input = json.load(open(os.path.join(self.data_set_loc, 'task_3_details.txt')))
        self.movieLSH = MovieLSH(task_3_input["num_layers"], task_3_input["num_hashs"])
        (self.query_df, self.query_vector) = self.fetch_query_vector_from_csv()

        self.movieLSH.create_index_structure(task_3_input["movie_list"])

    def fetch_query_vector_from_csv(self):
        if os.path.isfile(self.data_set_loc + "/relevance-feedback-query-vector.csv"):
            df = self.data_extractor.get_relevance_feedback_query_vector()
        else:
            df = pd.DataFrame(columns=["latent-semantic-number-" + str(num) for num in range(1, 501)])
            zero_query_vector = {}
            for num in range(1, 501):
                zero_query_vector["latent-semantic-number-" + str(num)] = 0
            df = df.append(zero_query_vector, ignore_index=True)

        return df, df.values[-1]

    def save_query_vector_to_csv(self):
        new_query_point_dict = {}
        for num in range(1, 501):
            new_query_point_dict["latent-semantic-number-" + str(num)] = self.query_vector[num - 1]

        self.query_df = self.query_df.append(new_query_point_dict, ignore_index=True)
        self.query_df.to_csv(self.data_set_loc + "/relevance-feedback-query-vector.csv", index=False)

    def get_movie_tag_matrix(self):
        movie_tag_df = None
        try:
            movie_tag_df = self.data_extractor.get_movie_latent_semantics_data()
        except:
            print("Unable to find movie matrix for movies in latent space.\nAborting...")
            exit(1)

        movie_index = 0
        movie_ids_list = movie_tag_df.movieid
        for movie_id in movie_ids_list:
            self.movies_dict[movie_id] = movie_index
            movie_index += 1

        return movie_tag_df.values

    def get_feedback_data(self):
        data = None
        try:
            data = self.data_extractor.get_task4_feedback_data()
        except:
            print("Relevance feedback file is missing.\nAborting...")
            exit(1)

        return data

    def update_query_point(self):
        previous_query_vector = self.query_vector

        rel_query_vector = [0 for _ in range(1, 501)]
        irrel_query_vector = [0 for _ in range(1, 501)]

        feedback_data = self.get_feedback_data()
        for index, row in feedback_data.iterrows():
            movie_id = row['movie-id']
            relevancy = row['relevancy']
            if relevancy == 'relevant':
                for i in range(0, 500):
                    rel_query_vector[i] += self.movie_tag_matrix[self.movies_dict[movie_id]][i]
            elif relevancy == 'irrelevant':
                for i in range(0, 500):
                    irrel_query_vector[i] += self.movie_tag_matrix[self.movies_dict[movie_id]][i]

        relevant_data = feedback_data[feedback_data['relevancy'] == 'relevant']
        num_of_rel_movie_records = len(relevant_data['relevancy'])
        irrelevant_data = feedback_data[feedback_data['relevancy'] == 'irrelevant']
        num_of_irrel_movie_records = len(irrelevant_data['relevancy'])

        new_query_vector = []
        for i in range(0, 500):
            new_query_vector.append(
                previous_query_vector[i] + (rel_query_vector[i] / float(num_of_rel_movie_records)) - (
                irrel_query_vector[i] / float(num_of_irrel_movie_records)))

        self.query_vector = new_query_vector
        self.save_query_vector_to_csv()

    def get_nearest_neighbours(self, n):
        self.update_query_point()
        movie_ids = self.movieLSH.query_for_nearest_neighbours_using_csv(self.query_vector, n)

        return movie_ids

    def print_movie_recommendations_and_collect_feedback(self, n):
        nearest_movie_ids = self.get_nearest_neighbours(n)
        self.util.print_movie_recommendations_and_collect_feedback(nearest_movie_ids, 4, None)


if __name__ == "__main__":
    nn_rel_feed = NearestNeighbourBasedRelevanceFeedback()
    while True:
        n = int(input("\n\nEnter value of 'r' : "))
        nn_rel_feed.print_movie_recommendations_and_collect_feedback(n)
        confirmation = input("\n\nDo you want to continue? (y/Y/n/N): ")
        if confirmation != "y" and confirmation != "Y":
            break
