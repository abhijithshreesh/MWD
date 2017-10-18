import logging

import numpy
import pandas as pd
from scripts.phase2.common.task_2 import GenreTag
from sklearn.decomposition import PCA

from scripts.phase2.common.config_parser import ParseConfig

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class PcaGenreTag(GenreTag):
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
        genre_tag_tfidf_df.to_csv('genre_tag_matrix.csv', index = True , encoding='utf-8')


        # U, s, Vh = numpy.linalg.svd(genre_tag_tfidf_df.values,full_matrices=True)
        # a = 1
        #
        # x = genre_tag_tfidf_df.values
        # pca = PCA(n_components = 4)
        # pca.fit(x)
        # print(pca.explained_variance_ratio_)
        # print(pca.components_)

        df = pd.DataFrame(pd.read_csv('genre_tag_matrix.csv'))
        df1 = df.values
        #print(df1)
        # print(df.shape)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        df_sc = sc.fit_transform(df1[:, 1:])

        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        df_pca = pca.fit_transform(df_sc)
        explained_variance = pca.explained_variance_ratio_
        return (df_pca, explained_variance)

if __name__ == "__main__":
    obj = PcaGenreTag()
    (components, explained_variance) = obj.genre_tag(genre="Action")
    print (components)
    print (explained_variance)
