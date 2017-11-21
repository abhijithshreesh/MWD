import operator
import numpy
import pandas as pd

import config_parser
import data_extractor
from phase1_task_2 import GenreTag
from util import Util


class UserMovieRecommendation(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.mltags = self.data_extractor.get_mltags_data()
        self.genome_tags = self.data_extractor.get_genome_tags_data()
        self.combined_data = self.get_combined_data()
        self.util = Util()
        self.genre_tag = GenreTag()
        self.genre_data = self.genre_tag.get_genre_data()
        self.ordered_genre = []
        self.ordered_movie_names = []
        self.ordered_tag = []

    def get_all_movies_for_user(self, user_id):
        """
        Obtain all movies watched by the user
        :param user_id:
        :return: list of movies watched by the user
        """
        user_data = self.combined_data[self.combined_data['userid'] == user_id]
        user_data = user_data.sort_values('timestamp', ascending=False)
        movies = user_data['moviename'].unique()

        return movies

    def get_combined_data(self):
        """
        The data set under consideration for movie recommendation
        :return: dataframe which combines all the necessary fields needed for the recommendation system
        """
        result = self.mltags.merge(self.mlmovies, left_on="movieid", right_on="movieid", how="left")
        merged_result = result.merge(self.genome_tags, left_on="tagid", right_on="tagId", how="left")
        merged_result["tag_string"] = pd.Series(
            [str(tag) for tag in merged_result.tag],
            index=merged_result.index)
        del merged_result['tagid']
        del merged_result['tagId']
        del merged_result['year']
        del merged_result['genres']

        return merged_result

    def get_movie_tag_matrix(self):
        """
        Function to get movie_tag matrix containing list of tags in each movie
        :return: movie_tag_matrix
        """
        tag_df = self.genre_data
        tag_df["tag_string"] = pd.Series(
            [str(tag) for tag in tag_df.tag],
            index=tag_df.index)
        unique_tags = tag_df.tag_string.unique()
        idf_data = tag_df.groupby(['movieid'])['tag_string'].apply(set)
        tf_df = tag_df.groupby(['movieid'])['tag_string'].apply(list).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag_string))
        tf_weight_dict = {movie: self.genre_tag.assign_tf_weight(tags) for movie, tags in
                          list(movie_tag_dict.items())}
        idf_weight_dict = self.genre_tag.assign_idf_weight(idf_data, unique_tags)
        tag_df = self.genre_tag.get_model_weight(tf_weight_dict, idf_weight_dict, tag_df, 'tfidf')
        tag_df["total"] = tag_df.groupby(['movieid','tag_string'])['value'].transform('sum')
        temp_df = tag_df[["moviename", "tag_string", "total"]].drop_duplicates().reset_index()
        genre_tag_tfidf_df = temp_df.pivot_table('total', 'moviename', 'tag_string')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)

        return genre_tag_tfidf_df

    def get_movie_movie_matrix(self, model):
        """
        Finds movie_tag matrix and returns movie_movie_similarity matrix
        :param model:
        :return: movie_movie_similarity matrix
        """
        if model == "LDA":
            movie_tag_data_frame = self.combined_data
            tag_df = movie_tag_data_frame.groupby(['moviename'])['tag_string'].apply(list).reset_index()
            movies = tag_df.moviename.tolist()
            movies_tags_list = list(tag_df.tag_string)
            (U, Vh) = self.util.LDA(movies_tags_list, num_topics=10, num_features=len(self.combined_data.tag_string.unique()))
            movie_latent_matrix = self.util.get_doc_topic_matrix(U, num_docs=len(movies), num_topics=10)
        elif model == "SVD" or model == "PCA":
            movie_tag_frame = self.get_movie_tag_matrix()
            movie_tag_matrix = movie_tag_frame.values
            movies = list(movie_tag_frame.index.values)
            if model == "SVD":
                (U, s, Vh) = self.util.SVD(movie_tag_matrix)
                movie_latent_matrix = U[:, :10]
            else:
                (U, s, Vh) = self.util.PCA(movie_tag_matrix)
                tag_latent_matrix = U[:, :10]
                movie_latent_matrix = numpy.dot(movie_tag_matrix, tag_latent_matrix)
        elif model == "TD":
            tensor = self.fetch_movie_genre_tag_tensor()
            factors = self.util.CPDecomposition(tensor, 10)
            movies = self.ordered_movie_names
            movie_latent_matrix = factors[0]
        latent_movie_matrix = movie_latent_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_latent_matrix, latent_movie_matrix)

        return (movies, movie_movie_matrix)

    def compute_pagerank(self):
        """
        Function to prepare data for pageRank and calling pageRank method
        :return: list of (movie,weight) tuple
        """
        movie_tag_frame = self.get_movie_tag_matrix()
        movie_tag_matrix = movie_tag_frame.values
        movies = list(movie_tag_frame.index.values)
        tag_movie_matrix = movie_tag_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_tag_matrix, tag_movie_matrix)
        seed_movies = self.get_all_movies_for_user(user_id)

        return self.util.compute_pagerank(seed_movies, movie_movie_matrix, movies)

    def get_recommendation(self, user_id, model):
        """
        Function to recommend movies for a given user_id based on the given model
        :param user_id:
        :param model:
        :return: list of movies for the given user as a recommendation
        """
        watched_movies = self.get_all_movies_for_user(user_id)
        recommended_movies = []
        if len(watched_movies) == 0:
            print("THIS USER HAS NOT WATCHED ANY MOVIE")
            exit(1)
        if model == "PageRank":
            recommended_dict = self.compute_pagerank()
            for movie_p, weight_p in recommended_dict:
                if len(recommended_movies) == 5:
                    break
                if movie_p not in watched_movies:
                    recommended_movies.append(movie_p)
        elif model == "SVD" or model == "PCA" or model == "LDA" or model == "TD":
            (movies, movie_movie_matrix) = self.get_movie_movie_matrix(model)
            movie_row_dict = {}
            for i in range(0, len(movies)):
                if movies[i] in watched_movies:
                    movie_row_dict[self.util.get_movie_id(movies[i])] = movie_movie_matrix[i]
            distribution_list = self.util.get_distribution_count(watched_movies, 5)
            index = 0
            for movie in watched_movies:
                movie_row = movie_row_dict[self.util.get_movie_id(movie)]
                labelled_movie_row = dict(zip(movies, movie_row))
                num_of_movies_to_pick = distribution_list[index]
                # Remove the movies which are already watched
                for each in watched_movies:
                    del labelled_movie_row[each]
                # Remove the movies which are already in recommendation_list
                for each in recommended_movies:
                    del labelled_movie_row[each]
                labelled_movie_row_sorted = sorted(labelled_movie_row.items(), key=operator.itemgetter(1), reverse=True)
                labelled_movie_row_sorted = labelled_movie_row_sorted[0:num_of_movies_to_pick]
                for (m,v) in labelled_movie_row_sorted:
                    recommended_movies.append(m)
                if len(recommended_movies) == 5:
                    break
                index += 1

        return recommended_movies

    def fetch_movie_genre_tag_tensor(self):
        """
        Create Movie Genre Tag tensor
        :return: tensor
        """
        movie_list = self.genre_data["moviename"]
        movie_count = 0
        movie_dict = {}
        for element in movie_list:
            if element in movie_dict.keys():
                continue
            movie_dict[element] = movie_count
            movie_count += 1
            self.ordered_movie_names.append(element)

        genre_list = self.genre_data["genre"]
        genre_count = 0
        genre_dict = {}
        for element in genre_list:
            if element in genre_dict.keys():
                continue
            genre_dict[element] = genre_count
            genre_count += 1
            self.ordered_genre.append(element)

        tag_list = self.genre_data["tag"]
        tag_count = 0
        tag_dict = {}
        for element in tag_list:
            if element in tag_dict.keys():
                continue
            tag_dict[element] = tag_count
            tag_count += 1
            self.ordered_tag.append(element)

        tensor = numpy.zeros((movie_count, genre_count, tag_count))

        for index, row in self.genre_data.iterrows():
            movie = row["moviename"]
            genre = row["genre"]
            tag = row["tag"]
            movie_name = movie_dict[movie]
            genre_name = genre_dict[genre]
            tag_name = tag_dict[tag]
            tensor[movie_name][genre_name][tag_name] = 1

        return tensor


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_2_task_1a.py 146',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    user_id = 11613
    model = "SVD" # SVD,PCA,LDA,TD,PageRank
    obj = UserMovieRecommendation()
    recommended_movies = obj.get_recommendation(user_id=user_id, model=model)
    obj.util.print_movie_recommendations_and_collect_feedback(recommended_movies, 2, user_id)