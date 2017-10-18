import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_2 import GenreTag
import argparse
import scipy
import numpy
from sklearn.decomposition import TruncatedSVD
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

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
        U, s, Vh = numpy.linalg.svd(genre_tag_tfidf_df.values,full_matrices=True)

        df1 = genre_tag_tfidf_df.values
        # svd = TruncatedSVD(n_components = 4)
        # svd.fit(x)
        # print(svd.explained_variance_ratio_)
        # print(svd.components_)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        df_sc = sc.fit_transform(df1[:, 1:])

        # Applying PCA
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=4)
        df_svd = svd.fit_transform(df_sc)
        explained_variance = svd.explained_variance_ratio_

        return (df_svd, explained_variance)

if __name__ == "__main__":
    obj = SvdGenreTag()
    (df_svd, explained_variance) = obj.genre_tag(genre="Action")
    print (df_svd)
    print (explained_variance)