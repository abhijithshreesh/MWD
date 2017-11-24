import math

import config_parser
import data_extractor
from util import Util


class ProbabilisticRelevanceFeedbackUserMovieRecommendation(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.feedback_data = self.get_feedback_data()
        self.util = Util()
        self.movies_dict = {}
        self.tag_dict = {}
        self.movie_tag_matrix = self.get_movie_tag_matrix()
        self.feedback_metadata_dict = {}
        (self.relevant_tag_indices, self.relevant_movies) = self.get_relevant_tag_indices_and_movies()

    def get_relevant_tag_indices_and_movies(self):
        user_relevancy_info = self.feedback_data[self.feedback_data["user-id"] == self.user_id]
        movies = user_relevancy_info["movie-name"].unique()

        relevant_tag_indices = set()
        relevant_tags = set()
        for movie in movies:
            movie_tags = self.util.get_tag_list_for_movie(movie)
            for tag in movie_tags:
                relevant_tag_indices.add(self.tag_dict[tag])
                relevant_tags.add(tag)

        relevant_movies = set()
        for tag in relevant_tags:
            tag_movies = self.util.get_movies_for_tag(tag)
            for movie in tag_movies:
                relevant_movies.add(movie)

        watched_movies = self.util.get_all_movies_for_user(self.user_id)
        relevant_movies = set(relevant_movies) - set(watched_movies)

        return list(relevant_tag_indices), list(relevant_movies)

    def get_feedback_data(self):
        data = None
        try:
            data = self.data_extractor.get_task2_feedback_data()
        except:
            print("Relevance feedback information missing.\nAborting...")
            exit(1)

        return data

    def get_movie_tag_matrix(self):
        movie_tag_matrix = self.util.get_movie_tag_matrix()
        movie_tag_matrix[movie_tag_matrix > 0] = 1

        movie_index = 0
        movies_list = list(movie_tag_matrix.index.values)
        for movie in movies_list:
            self.movies_dict[movie] = movie_index
            movie_index += 1

        tag_index = 0
        tags_list = list(movie_tag_matrix.columns.values)
        for tag in tags_list:
            self.tag_dict[tag] = tag_index
            tag_index += 1

        return movie_tag_matrix.values

    def get_movie_similarity(self, movie_name):
        movie_index = self.movies_dict[movie_name]
        movie_tag_values = self.movie_tag_matrix[movie_index]

        similarity = 0
        for tag in self.relevant_tag_indices:
            if tag in self.feedback_metadata_dict.keys():
                (p_i, u_i) = self.feedback_metadata_dict[tag]
            else:
                (p_i, u_i) = self.get_feedback_metadata(tag)
                self.feedback_metadata_dict[tag] = (p_i, u_i)
            numerator = p_i * (1 - u_i)
            denominator = u_i * (1 - p_i)
            temp = movie_tag_values[tag] * (math.log(numerator / denominator))
            similarity += temp

        return similarity

    def get_feedback_metadata(self, tag_index):
        user_feedback_data = self.feedback_data[self.feedback_data['user-id'] == self.user_id]
        user_relevant_data = user_feedback_data[user_feedback_data['relevancy'] == 'relevant']
        user_relevant_movies = user_relevant_data['movie-name'].unique()
        user_irrelevant_data = user_feedback_data[user_feedback_data['relevancy'] == 'irrelevant']
        user_irrelevant_movies = user_irrelevant_data['movie-name'].unique()

        R = len(user_relevant_movies)
        N = R + len(user_irrelevant_movies)

        count = 0
        for movie in user_relevant_movies:
            movie_index = self.movies_dict[movie]
            if self.movie_tag_matrix[movie_index][tag_index] == 1:
                count += 1
        r_i = count

        count = 0
        for movie in user_feedback_data['movie-name'].unique():
            movie_index = self.movies_dict[movie]
            if self.movie_tag_matrix[movie_index][tag_index] == 1:
                count += 1
        n_i = count

        numerator = r_i + 0.5
        denominator = R + 1
        p_i = numerator / float(denominator)

        numerator = n_i - r_i + 0.5
        denominator = N - R + 1
        u_i = numerator / float(denominator)

        return p_i, u_i

    def get_movie_recommendations(self):
        movie_similarity = {}

        for movie in self.relevant_movies:
            movie_similarity[movie] = self.get_movie_similarity(movie)

        movie_recommendations = []
        for movie in sorted(movie_similarity, key=movie_similarity.get, reverse=True):
            if len(movie_recommendations) == 5:
                break
            movie_recommendations.append(movie)

        return movie_recommendations

    def print_movie_recommendations_and_collect_feedback(self):
        movies = self.get_movie_recommendations()
        self.util.print_movie_recommendations_and_collect_feedback(movies, 2, self.user_id)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_3_task_2.py user_id',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    user_id = 20
    prop_rel_feed_rec = ProbabilisticRelevanceFeedbackUserMovieRecommendation(user_id)
    prop_rel_feed_rec.print_movie_recommendations_and_collect_feedback()
