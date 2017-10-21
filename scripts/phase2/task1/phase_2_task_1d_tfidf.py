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
import numpy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()
util = Util()
genre_tag = GenreTag()

class SimilarActorsFromDiffMovies(ActorActorMatrix):

    def __init__(self):
        """
        Initialiazing the data extractor object to get data from the csv files
        """
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def get_actors_of_movie(self, moviename):
        actor_movie_table = self.data_extractor.get_movie_actor_data()
        movieid = util.get_movie_id(moviename)
        actor_movie_table = actor_movie_table[actor_movie_table['movieid']== movieid]
        actorids = actor_movie_table["actorid"].tolist()
        return actorids

    def get_movie_tag_matrix(self):
        data_frame = genre_tag.get_genre_data()
        tag_df = data_frame.reset_index()
        #temp_df = data_frame[data_frame["genre"] == genre]
        unique_tags = tag_df.tag.unique()
        idf_data = tag_df.groupby(['movieid'])['tag'].apply(set)
        tf_df = tag_df.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag))
        tf_weight_dict = {movie: genre_tag.assign_tf_weight(tags.split(',')) for movie, tags in
                          list(movie_tag_dict.items())}
        idf_weight_dict = {}
        idf_weight_dict = genre_tag.assign_idf_weight(idf_data, unique_tags)
        tag_df = genre_tag.get_model_weight(tf_weight_dict, idf_weight_dict, tag_df, 'tfidf')
        tag_df["total"] = tag_df.groupby(['movieid','tag'])['value'].transform('sum')
        #tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
        # actor_tag_dict = dict(zip(tag_df.tag, tag_df.total))
        temp_df = tag_df[["moviename", "tag", "total"]].drop_duplicates().reset_index()



        genre_tag_tfidf_df = temp_df.pivot_table('total', 'moviename', 'tag')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)
        genre_tag_tfidf_df.to_csv('movie_tag_matrix1d.csv', index=True, encoding='utf-8')
        return genre_tag_tfidf_df



    def get_movie_movie_vector(self, moviename):

        movie_tag_frame = self.get_movie_tag_matrix()
        movie_tag_matrix = movie_tag_frame.values
        movies = list(movie_tag_frame.index.values)
        tags = list(movie_tag_frame)
        tag_movie_matrix = movie_tag_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_tag_matrix, tag_movie_matrix)

        # (matrix, actorids) = self.fetchActorActorSimilarityMatrix()
        # #In the pre-processing task above command should be run so that actor_actor_similarity matrix will be generated
        # #and saved as csv which can be used multiple number of times. Will comment the above line, when its done.
        #
        # # Loading the required actor_actor_similarity matrix from csv
        # df = pd.DataFrame(pd.read_csv('actor_actor_matrix.csv', header=None))
        # matrix = df.values
        #
        # actorids = util.get_sorted_actor_ids()
        #
        index_movie = None
        for i,j in enumerate(movies):
            if j == moviename:
                index_movie = i
                break

        if index_movie==None:
            print ("Movie Id not found.")
            return None

        movie_row = movie_movie_matrix[index_movie].tolist()
        movie_movie_dict = dict(zip(movies, movie_row))
        del movie_movie_dict[moviename]
        movie_movie_dict = sorted(movie_movie_dict.items(), key=operator.itemgetter(1), reverse=True)
        return movie_movie_dict

    def most_similar_actors(self, moviename):
        movieid = util.get_movie_id(moviename)
        movie_movie_dict = self.get_movie_movie_vector(moviename)
        if movie_movie_dict == None:
            return None
        actors = []
        for (movie,val) in movie_movie_dict:
            if val <= 0:
                break
            movieid = util.get_movie_id(movie)
            actors = actors + self.get_actors_of_movie(movie)
            if len(actors) >= 10:
                break

        actors_of_given_movie = self.get_actors_of_movie(moviename)

        actorsFinal = [x for x in actors if x not in actors_of_given_movie]

        actornames = []
        for actorid in actorsFinal:
            actor = util.get_actor_name_for_id(actorid)
            actornames.append(actor)

        return actornames


if __name__ == "__main__":
    obj = SimilarActorsFromDiffMovies()
    #actor_actor_dict = obj.get_actor_actor_vector(actorid=542238)
    #obj.get_movie_tag_matrix()
    #obj.get_actors_of_movie(moviename="Hannibal")
    #movie_movie_dict = obj.get_movie_movie_vector(movieid="Hannibal")
    actors = obj.most_similar_actors(moviename="Swordfish")
    print (actors)