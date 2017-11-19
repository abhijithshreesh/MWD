import config_parser
import data_extractor
import numpy
from phase1_task_2 import GenreTag
from util import Util


class UserMovieRecommendation(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.mltags = self.data_extractor.get_mltags_data()
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.mlratings = self.data_extractor.get_mlratings_data()
        self.combined_data = self.get_combined_data()
        self.users = self.data_extractor.get_mlusers_data()
        self.util = Util()
        self.genre_tag = GenreTag()

    def get_all_movies_for_user(self, user_id):
        """
        Obtain all movies watched by the user
        :param user_id:
        :return: list of movies watched by the user
        """
        user_data = self.combined_data[self.combined_data['userid'] == user_id]
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
            data_frame = self.data_extractor.get_mlmovies_data()
            tag_data_frame = self.data_extractor.get_genome_tags_data()
            movie_data_frame = self.data_extractor.get_mltags_data()
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
        latent_movie_matrix = movie_latent_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_latent_matrix, latent_movie_matrix)
        return (movies, movie_movie_matrix)

    def compute_pagerank(self):
        movie_tag_frame = self.get_movie_tag_matrix()
        movie_tag_matrix = movie_tag_frame.values
        movies = list(movie_tag_frame.index.values)
        tags = list(movie_tag_frame)
        tag_movie_matrix = movie_tag_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_tag_matrix, tag_movie_matrix)
        seed_movies = self.get_all_movies_for_user(user_id)
        return self.util.compute_pagerank(seed_movies, movie_movie_matrix, movies)

    def tensor_decompose(self, model):
        return (None, None)

    def get_result(self, user_id, model):
        """
        This method is yet to fully implemented.
        :param model:
        :return: List of recommended movies
        """
        watched_movies = self.get_all_movies_for_user(user_id)
        if watched_movies == None:
            print("THIS USER HAS NOT WATCHED ANY MOVIE")
            exit(1)
        if model == "PageRank":
            recommended_movies = self.compute_pagerank()
            print(recommended_movies)
        elif model == "SVD" or model == "PCA" or model == "LDA":
            (movies, movie_movie_matrix) = self.get_movie_movie_matrix(model)
        elif model == "TD":
            (movies, movie_movie_matrix) = self.tensor_decompose(model)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_2_task_1a.py 146',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    user_id = 11824
    obj = UserMovieRecommendation()
    obj.get_result(user_id=user_id, model="LDA")