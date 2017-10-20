import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_2 import GenreTag
import argparse
import scipy
import numpy
from sklearn.decomposition import TruncatedSVD
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from scripts.phase2.common.util import Util

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()
util = Util()

class SvdGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def genre_tag(self, genre):
        """
        Triggers the compute function and outputs the result tag vector
        :param genre:
        :param model:
        :return: returns a dictionary of Genres to dictionary of tags and weights.
        """

        genre_tag_frame = self.get_genre_data()
        given_genre_frame = self.combine_computed_weights(genre_tag_frame, "TFIDF", genre)
        temp_df = given_genre_frame[["moviename", "tag", "total"]].drop_duplicates()
        genre_tag_tfidf_df = temp_df.pivot(index='moviename', columns='tag', values='total')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)

        genre_tag_tfidf_df.to_csv('genre_tag_matrix.csv', index=True, encoding='utf-8')

        df = pd.DataFrame(pd.read_csv('genre_tag_matrix.csv'))
        df1 = df.values[:, 1:]
        row_headers = list(df["moviename"])
        column_headers = list(df)
        del column_headers[0]

        (U, s, Vh) = util.SVD(df1)

        # To print latent semantics
        latents = util.get_latent_semantics(5, Vh)
        util.print_latent_semantics(latents, column_headers)

        u_frame = pd.DataFrame(U[:,:5], index=row_headers)
        v_frame = pd.DataFrame(Vh[:5,:], columns=column_headers)
        u_frame.to_csv('u_1a_svd.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1a_svd.csv', index=True, encoding='utf-8')
        return (u_frame, v_frame, s)

if __name__ == "__main__":
    obj = SvdGenreTag()
    (u_frame, v_frame, s) = obj.genre_tag(genre="Action")