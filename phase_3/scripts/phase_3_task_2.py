import math

import config_parser
import data_extractor
import numpy as np


class ProbabilisticRelevanceFeedbackUserMovieRecommendation(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.mltags = self.data_extractor.get_mltags_data()
        self.genome_tags = self.data_extractor.get_genome_tags_data()
        self.combined_data = self.get_combined_data()
        self.movies_dict = {}
        self.tags_dict = {}
        self.movie_tag_matrix = self.get_movie_tag_matrix()

    def get_combined_data(self):
        """
        The data set under consideration for movie recommendation
        :return: dataframe which combines all the necessary fields needed for the recommendation system
        """
        temp = self.mltags.merge(self.mlmovies, left_on="movieid", right_on="movieid", how="left")
        result = temp.merge(self.genome_tags, left_on="tagid", right_on="tagId", how="left")
        del result['timestamp']
        del result['tagid']
        del result['tagId']
        del result['year']
        del result['movieid']
        del result['genres']

        return result

    def get_movie_tag_matrix(self):
        tags = self.combined_data['tag'].unique()
        tags.sort()
        tag_index = 0
        tag_dict = {}
        for element in tags:
            tag_dict[element] = tag_index
            tag_index += 1
        self.tags_dict = tag_dict

        movies = self.combined_data['moviename'].unique()
        movies.sort()
        movieid_index = 0
        movieid_dict = {}
        for element in movies:
            movieid_dict[element] = movieid_index
            movieid_index += 1
        self.movies_dict = movieid_dict

        movie_tag_matrix = np.zeros((movieid_index, tag_index))
        for index, row in self.combined_data.iterrows():
            tag_index = self.tags_dict[row['tag']]
            movie_index = self.movies_dict[row['moviename']]
            movie_tag_matrix[movie_index][tag_index] = 1

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
            similarity += temp

        return similarity

    def get_feedback_metadata(self, tag_index):
        try:
            self.feedback_data = self.data_extractor.get_task2_feedback_data()
        except:
            print("Relevance feedback information missing.\nAborting...")
            exit(1)

        user_feedback_data = self.feedback_data[self.feedback_data['user_id'] == self.user_id]
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

        count = 0
        movie_recommendations = []
        for movie in sorted(movie_similarity, key=movie_similarity.get, reverse=True):
            if count == 5:
                break
            movie_recommendations.append(movie)
            count += 1

        return movie_recommendations


if __name__ == "__main__":
    user_id = 3
    prop_rel_feed_rec = ProbabilisticRelevanceFeedbackUserMovieRecommendation(user_id)
    movies = prop_rel_feed_rec.get_movie_recommendations()
    print("Movie recommendations for user id " + str(user_id) + ": ")
    for movie in movies:
        print(movie)
