import logging
from scripts.phase2.common.config_parser import ParseConfig
import numpy
from sklearn.decomposition import PCA
import logging

import numpy
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
from scripts.phase2.task1.phase_2_task_1b_svd import SvdGenreActor

log = logging.getLogger(__name__)
conf = ParseConfig()

class PcaGenreActor(SvdGenreActor):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def pca_genre_actor(self, genre):
        """
        Triggers the compute function and outputs the result tag vector
        :param genre:
        :param model:
        :return: returns a dictionary of Genres to dictionary of tags and weights.
        """

        genre_actor_frame = self.get_genre_actor_data_frame()
        rank_weight_dict = self.assign_rank_weight(genre_actor_frame[['movieid', 'actor_movie_rank']])
        genre_actor_frame = self.combine_computed_weights(genre_actor_frame, rank_weight_dict, "TFIDF", genre)
        temp_df = genre_actor_frame[["movieid", "actorid_string", "total"]].drop_duplicates()
        genre_actor_tfidf_df = temp_df.pivot(index='movieid', columns='actorid_string', values='total')
        genre_actor_tfidf_df = genre_actor_tfidf_df.fillna(0)
        genre_actor_tfidf_df.to_csv('genre_actor_matrix.csv', index = False , encoding='utf-8')
        U, s, Vh = numpy.linalg.svd(genre_actor_tfidf_df.values,full_matrices=True)
        a = 1

        x = genre_actor_tfidf_df.values
        pca = PCA(n_components = 4)
        pca.fit(x)
        print(pca.explained_variance_ratio_)
        print(pca.components_)

if __name__ == "__main__":
    obj = PcaGenreActor()
    obj.pca_genre_actor(genre="Action")
