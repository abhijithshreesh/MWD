import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_2 import GenreTag
from sklearn.preprocessing import Imputer
from sklearn.decomposition import LatentDirichletAllocation
import argparse
from collections import Counter
from gensim import corpora, models
import gensim
import numpy
import operator
from scripts.phase2.common.data_extractor import DataExtractor
from scripts.phase2.common.util import Util
from scripts.phase2.task1.phase_2_task_1d_tfidf import SimilarActorsFromDiffMovies

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()
sim_act_diff_mov_tf = SimilarActorsFromDiffMovies()

class SimilarActorsFromDiffMoviesLda(object):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.util = Util()

    def most_similar_actors_lda(self, moviename):
        data_frame = self.data_extractor.get_mlmovies_data()
        tag_data_frame = self.data_extractor.get_genome_tags_data()
        movie_data_frame = self.data_extractor.get_mltags_data()
        movie_tag_data_frame = movie_data_frame.merge(tag_data_frame, how="left", left_on="tagid", right_on="tagId")
        movie_tag_data_frame = movie_tag_data_frame.merge(data_frame, how="left", left_on="movieid", right_on="movieid")
        tag_df = movie_tag_data_frame.groupby(['movieid'])['tag'].apply(list).reset_index()

        tag_df = tag_df.sort_values('movieid')
        movies = tag_df.moviename.tolist()
        tag_df = list(tag_df.iloc[:, 1])

        (U, Vh) = self.util.LDA(tag_df, num_topics=5, num_features=1000)

        movie_topic_matrix = self.util.get_doc_topic_matrix(U, num_docs=len(movies), num_topics=5)
        topic_movie_matrix = movie_topic_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_topic_matrix,topic_movie_matrix)

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
        for (movie, val) in movie_movie_dict:
            if val <= 0:
                break
            movieid = self.util.get_movie_id(movie)
            actors = actors + sim_act_diff_mov_tf.get_actors_of_movie(movie)
            if len(actors) >= 10:
                break

        actors_of_given_movie = sim_act_diff_mov_tf.get_actors_of_movie(moviename)

        actorsFinal = [x for x in actors if x not in actors_of_given_movie]

        actornames = []
        for actorid in actorsFinal:
            actor = self.util.get_actor_name_for_id(actorid)
            actornames.append(actor)

        return actornames


if __name__ == "__main__":
    obj = SimilarActorsFromDiffMoviesLda()
    movie_movie_dict = obj.most_similar_actors_lda(moviename='Swordfish')
    print(movie_movie_dict)