import pandas as pd
import logging
from scripts.config_parser import ParseConfig
from scripts.task_2 import GenreTag
import argparse
import scipy
import numpy
from gensim import corpora, models
import gensim
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class LdaGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def get_lda_data(self, genre):
        data_frame = self.get_genre_data().reset_index()
        genre_data_frame = data_frame[data_frame["genre"]==genre]
        tag_df = genre_data_frame
        tf_df = genre_data_frame.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag))
        tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in
                          list(movie_tag_dict.items())}
        #tag_df = self.get_model_weight(tf_weight_dict, {}, genre_data_frame, "TF")
        tag_df["value"] = pd.Series([tf_weight_dict.get(movieid, 0).get(tag, 0) for index, tag, movieid
                            in zip(tag_df.index, tag_df.tag, tag_df.movieid)], index=tag_df.index)
        tag_df["total"] = tag_df.groupby(['movieid', 'tag'])['value'].transform('sum')
        tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
        a=1

        #ldamodel = gensim.models.ldamodel.LdaModel(tag_df.values, num_topics=3)
        #print(ldamodel.print_topics(num_topics=3, num_words=3))

if __name__ == "__main__":
    obj = LdaGenreTag()
    obj.get_lda_data(genre="Action")
