import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
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

class LdaGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")


    def get_tag_count(self, tag_series):
        counter = Counter()
        for each in tag_series:
            counter[each] += 1
        return dict(counter)

    def get_lda_data(self, genre):
        data_frame = self.get_genre_data().reset_index()
        genre_data_frame = data_frame[data_frame["genre"]==genre]
        tag_df = genre_data_frame.groupby(['movieid'])['tag'].apply(list).reset_index()
        tag_df = tag_df.sort_values('movieid')
        #tag_df.to_csv('movie_tag_lda.csv', index=True, encoding='utf-8')
        #movieid_list = tag_df.movieid.tolist()
        #tag_matrix = tag_df[["tag"]].values
        #tag_matrix = list(tag_matrix.iloc[:,1])
        tag_df = list(tag_df.iloc[:,1])

        (U, Vh) = util.LDA(tag_df, num_topics=4, num_features=1000)

        for latent in Vh:
            print(latent)

        # for doc in U:
        #     print(doc)

if __name__ == "__main__":
    obj = LdaGenreTag()
    lda_comp = obj.get_lda_data(genre="Action")
    print (lda_comp)