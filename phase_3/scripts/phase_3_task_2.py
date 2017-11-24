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
        self.movies_dict = {}
        self.util = Util()
        self.movie_tag_matrix = self.get_movie_tag_matrix()

    def get_movie_tag_matrix(self):
        movie_tag_matrix = self.util.get_movie_tag_matrix()
        movie_tag_matrix[movie_tag_matrix > 0] = 1

        return movie_tag_matrix

    def get_movie_similarity(self, movie_name):
        movie_index = self.movies_dict[movie_name]
        movie_tag_values = self.movie_tag_matrix[movie_index]

        similarity = 0
        tag_index = 0
        for tag in movie_tag_values:
            (p_i, u_i) = self.get_feedback_metadata(tag_index)
            numerator = p_i * (1 - u_i)
            denominator = u_i * (1 - p_i)
            temp = tag * (math.log(numerator / denominator))
            tag_index += 1
            similarity += temp

        return similarity

    def get_feedback_metadata(self, tag_index):
        try:
            self.feedback_data = self.data_extractor.get_task2_feedback_data()
        except:
            print("Relevance feedback information missing.\nAborting...")
            exit(1)

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

        n_i_by_N = n_i / float(N)

        numerator = r_i + n_i_by_N
        denominator = R + 1
        p_i = numerator / float(denominator)

        numerator = n_i - r_i + n_i_by_N
        denominator = N - R + 1
        u_i = numerator / float(denominator)

        return p_i, u_i

    def get_movie_recommendations(self):
        movie_similarity = {}
        for movie in self.movies_dict.keys():
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
    user_id = 3
    prop_rel_feed_rec = ProbabilisticRelevanceFeedbackUserMovieRecommendation(user_id)
    prop_rel_feed_rec.print_movie_recommendations_and_collect_feedback()
