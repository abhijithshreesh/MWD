import operator
import numpy

import config_parser
import data_extractor
from phase1_task_2 import GenreTag
from util import Util
from tensor import MovieTagGenreTensor


class UserMovieRecommendation(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.mltags = self.data_extractor.get_mltags_data()
        self.combined_data = self.get_combined_data()
        self.util = Util()
        self.genre_tag = GenreTag()
        self.tensor = MovieTagGenreTensor()

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
        return result

    def get_movie_tag_matrix(self):
        """
        Function to get movie_tag matrix containing list of tags in each movie
        :return: movie_tag_matrix
        """
        data_frame = self.genre_tag.get_genre_data()
        tag_df = data_frame.reset_index()
        unique_tags = tag_df.tag.unique()
        idf_data = tag_df.groupby(['movieid'])['tag'].apply(set)
        tf_df = tag_df.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag))
        tf_weight_dict = {movie: self.genre_tag.assign_tf_weight(tags.split(',')) for movie, tags in
                          list(movie_tag_dict.items())}
        idf_weight_dict = {}
        idf_weight_dict = self.genre_tag.assign_idf_weight(idf_data, unique_tags)
        tag_df = self.genre_tag.get_model_weight(tf_weight_dict, idf_weight_dict, tag_df, 'tfidf')
        tag_df["total"] = tag_df.groupby(['movieid','tag'])['value'].transform('sum')
        temp_df = tag_df[["moviename", "tag", "total"]].drop_duplicates().reset_index()
        genre_tag_tfidf_df = temp_df.pivot_table('total', 'moviename', 'tag')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)

        return genre_tag_tfidf_df

    def get_movie_movie_matrix(self, model):
        """
        Finds movie_tag matrix and returns movie_movie_similarity matrix
        :param model:
        :return: movie_movie_similarity matrix
        """
        if model == "LDA":
            data_frame = self.mlmovies
            tag_data_frame = self.data_extractor.get_genome_tags_data()
            movie_data_frame = self.mltags
            movie_tag_data_frame = movie_data_frame.merge(tag_data_frame, how="left", left_on="tagid", right_on="tagId")
            movie_tag_data_frame = movie_tag_data_frame.merge(data_frame, how="left", left_on="movieid", right_on="movieid")
            tag_df = movie_tag_data_frame.groupby(['movieid'])['tag'].apply(list).reset_index()
            tag_df = tag_df.sort_values('movieid')
            movies = tag_df.movieid.tolist()
            movies = [self.util.get_movie_name_for_id(movieid) for movieid in movies]
            tag_df = list(tag_df.iloc[:, 1])
            (U, Vh) = self.util.LDA(tag_df, num_topics=10, num_features=1000)
            movie_latent_matrix = self.util.get_doc_topic_matrix(U, num_docs=len(movies), num_topics=10)
        elif model == "SVD" or model == "PCA":
            movie_tag_frame = self.get_movie_tag_matrix()
            movie_tag_matrix = movie_tag_frame.values
            movies = list(movie_tag_frame.index.values)
            tags = list(movie_tag_frame)
            if model == "SVD":
                (U, s, Vh) = self.util.SVD(movie_tag_matrix)
                movie_latent_matrix = U[:, :10]
            else:
                (U, s, Vh) = self.util.PCA(movie_tag_matrix)
                tag_latent_matrix = U[:, :10]
                movie_latent_matrix = numpy.dot(movie_tag_matrix, tag_latent_matrix)
        elif model == "TD":
            movies = self.tensor.ordered_movie_names
            movie_latent_matrix = self.tensor.factors[0]
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
        tags = list(movie_tag_frame)
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
            return recommended_movies
        elif model == "SVD" or model == "PCA" or model == "LDA" or model == "TD":
            (movies, movie_movie_matrix) = self.get_movie_movie_matrix(model)
            movie_row_dict = {}
            for i in range(0, len(movies)):
                if movies[i] in watched_movies:
                    movie_row_dict[self.util.get_movie_id(movies[i])] = movie_movie_matrix[i]
            distribution_list = self.get_distribution_count(watched_movies, 5)
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
                for key in labelled_movie_row.keys():
                    labelled_movie_row[key] = abs(labelled_movie_row[key])
                labelled_movie_row_sorted = sorted(labelled_movie_row.items(), key=operator.itemgetter(1), reverse=True)
                labelled_movie_row_sorted = labelled_movie_row_sorted[0:num_of_movies_to_pick]
                for (m,v) in labelled_movie_row_sorted:
                    recommended_movies.append(m)
                index += 1
            return recommended_movies

    def get_distribution_count(self, seed_nodes, num_of_seeds_to_recommend):
        """
        Given the number of seeds to be recommended and the seed_nodes,
        returns the distribution for each seed_node considering order
        :param seed_nodes:
        :param num_of_seeds_to_recommend:
        :return: distribution_list
        """
        seed_value = float(num_of_seeds_to_recommend / len(seed_nodes))
        seed_value_list = [seed_value for seed in seed_nodes]
        delta = seed_value / len(seed_nodes)
        for i in range(0, len(seed_nodes) - 1):
            seed_value_list[i] = round(seed_value_list[i] + (len(seed_nodes) - 1 - i) * delta)
            for j in range(i + 1, len(seed_nodes)):
                seed_value_list[j] = seed_value_list[j] - delta
        seed_value_list[len(seed_nodes) - 1] = round(seed_value_list[len(seed_nodes) - 1])
        total_count = 0
        for val in seed_value_list:
            total_count = total_count + val
        difference = num_of_seeds_to_recommend - total_count
        if(difference > 0):
            for i in range(0, len(seed_value_list)):
                if seed_value_list[i] == 0:
                    seed_value_list[i] = 1
                    difference -= 1
                    if difference == 0:
                        return seed_value_list
            for i in range(0, len(seed_value_list)):
                seed_value_list[i] += 1
                difference -= 1
                if difference == 0:
                    return seed_value_list
        elif(difference < 0):
            for i in range(0, len(seed_value_list)):
                if seed_value_list[len(seed_value_list) - 1 - i] != 0:
                    seed_value_list[len(seed_value_list) - 1 - i] -= 1
                    difference += 1
                if difference == 0:
                    return seed_value_list
        return seed_value_list


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_2_task_1a.py 146',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    user_id = 11613
    obj = UserMovieRecommendation()
    recommended_movies = obj.get_recommendation(user_id=user_id, model="SVD")
    print("SVD : ", recommended_movies)
    recommended_movies = obj.get_recommendation(user_id=user_id, model="PCA")
    print("PCA : ", recommended_movies)
    recommended_movies = obj.get_recommendation(user_id=user_id, model="LDA")
    print("LDA : ", recommended_movies)
    recommended_movies = obj.get_recommendation(user_id=user_id, model="TD")
    print("TD : ", recommended_movies)
    recommended_movies = obj.get_recommendation(user_id=user_id, model="PageRank")
    print("PageRank : ", recommended_movies)