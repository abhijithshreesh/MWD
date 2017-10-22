import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from scripts.phase2.common.task_2 import GenreTag
from scripts.phase2.common.util import Util
from sklearn.preprocessing import Imputer
from sklearn.decomposition import LatentDirichletAllocation
import argparse
from collections import Counter
from gensim import corpora, models
import gensim

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()
util = Util()

class LdaGenreActor(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)


    def get_tag_count(self, tag_series):
        counter = Counter()
        for each in tag_series:
            counter[each] += 1
        return dict(counter)

    def get_lda_data(self, genre):
        # Getting movie_genre_data
        movie_genre_data_frame = self.data_extractor.get_mlmovies_data()
        movie_genre_data_frame = self.split_genres(movie_genre_data_frame)

        # Getting actor_movie_data
        movie_actor_data_frame = self.data_extractor.get_movie_actor_data()

        genre_actor_frame = movie_genre_data_frame.merge(movie_actor_data_frame, how="left", left_on="movieid",
                                                         right_on="movieid")
        # genre_actor_frame = genre_actor_frame[genre_actor_frame['year'].notnull()].reset_index()
        genre_actor_frame = genre_actor_frame[["movieid", "year", "genre", "actorid", "actor_movie_rank"]]

        genre_actor_frame["actorid_string"] = pd.Series(
            [str(id) for id in genre_actor_frame.actorid],
            index=genre_actor_frame.index)

        genre_data_frame = genre_actor_frame[genre_actor_frame["genre"]==genre]
        actor_df = genre_data_frame.groupby(['movieid'])['actorid_string'].apply(list).reset_index()
        actor_df = actor_df.sort_values('movieid')
        actor_df.to_csv('movie_actor_lda.csv', index=True, encoding='utf-8')
        #movieid_list = tag_df.movieid.tolist()
        #tag_matrix = tag_df[["tag"]].values
        #tag_matrix = list(tag_matrix.iloc[:,1])
        actor_df = list(actor_df.iloc[:,1])

        (U, Vh) = util.LDA(actor_df, num_topics=4, num_features=1000)

        for latent in Vh:
            print(latent)

        # for doc in U:
        #     print(doc)

if __name__ == "__main__":
    obj = LdaGenreActor()
    lda_comp = obj.get_lda_data(genre="Action")
    #print (lda_comp)