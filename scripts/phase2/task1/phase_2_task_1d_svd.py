import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from collections import Counter
import math
import argparse
import operator
from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
from scripts.phase2.common.util import Util
from scripts.phase2.common.task_2 import GenreTag
from scripts.phase2.task1.phase_2_task_1d_tfidf import SimilarActorsFromDiffMovies
from scipy import linalg
from sklearn.preprocessing import StandardScaler
import numpy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()
util = Util()
genre_tag = GenreTag()
sim_act_diff_mov_tf = SimilarActorsFromDiffMovies()

class SimilarActorsFromDiffMoviesSvd(object):

    def __init__(self):
        """
        Initialiazing the data extractor object to get data from the csv files
        """
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def most_similar_actors_svd(self, moviename):
        movie_tag_frame = sim_act_diff_mov_tf.get_movie_tag_matrix()
        movie_tag_matrix = movie_tag_frame.values
        movies = list(movie_tag_frame.index.values)
        tags = list(movie_tag_frame)

        # # Feature Scaling
        # sc = StandardScaler()
        # df_sc = sc.fit_transform(movie_tag_matrix[:, :])
        #
        # # Calculating SVD
        # U, s, Vh = linalg.svd(df_sc)

        (U,s,Vh) = util.SVD(movie_tag_matrix)

        u_frame = pd.DataFrame(U[:, :5], index=movies)
        v_frame = pd.DataFrame(Vh[:5, :], columns=tags)
        u_frame.to_csv('u_1d_svd.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1d_svd.csv', index=True, encoding='utf-8')

        movie_latent_matrix = u_frame.values
        latent_movie_matrix = movie_latent_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_latent_matrix, latent_movie_matrix)

        index_movie = None
        for i, j in enumerate(movies):
            if j == moviename:
                index_movie = i
                break

        if index_movie == None:
            print("Movie Id not found.")
            return None

        movie_row = movie_movie_matrix[index_movie].tolist()
        movie_movie_dict = dict(zip(movies, movie_row))
        del movie_movie_dict[moviename]

        for key in movie_movie_dict.keys():
            movie_movie_dict[key] = abs(movie_movie_dict[key])

        movie_movie_dict = sorted(movie_movie_dict.items(), key=operator.itemgetter(1), reverse=True)

        if movie_movie_dict == None:
            return None
        actors = []
        for (movie,val) in movie_movie_dict:
            if val <= 0:
                break
            movieid = util.get_movie_id(movie)
            actors = actors + sim_act_diff_mov_tf.get_actors_of_movie(movie)
            if len(actors) >= 10:
                break

        actors_of_given_movie = sim_act_diff_mov_tf.get_actors_of_movie(moviename)

        actorsFinal = [x for x in actors if x not in actors_of_given_movie]

        actornames = []
        for actorid in actorsFinal:
            actor = util.get_actor_name_for_id(actorid)
            actornames.append(actor)

        return actornames

if __name__ == "__main__":
    obj = SimilarActorsFromDiffMoviesSvd()
    # actor_actor_dict = obj.get_actor_actor_vector(actorid=542238)
    # obj.get_movie_tag_matrix()
    # obj.get_actors_of_movie(moviename="Hannibal")
    # movie_movie_dict = obj.get_movie_movie_vector(movieid="Hannibal")
    actors = obj.most_similar_actors_svd(moviename="Swordfish")
    print(actors)
