import logging
from scripts.phase2.common.config_parser import ParseConfig
import numpy
from sklearn.decomposition import PCA
import logging
import pandas as pd
import numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from scripts.phase2.common.util import Util

logging.basicConfig(level=logging.INFO)
from scripts.phase2.task1.phase_2_task_1b_svd import SvdGenreActor

log = logging.getLogger(__name__)
conf = ParseConfig()
util = Util()

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

        df = pd.DataFrame(pd.read_csv('genre_actor_matrix.csv'))
        df1 = df.values[:, :]
        column_headers = list(df)
        #del column_headers[0]

        column_headers_names = []

        for col_head in column_headers:
            col_head_name = util.get_actor_name_for_id(int(col_head))
            column_headers_names = column_headers_names + [col_head_name]

        (U, s, Vh) = util.PCA(df1)

        # To print latent semantics
        latents = util.get_latent_semantics(5, Vh)
        util.print_latent_semantics(latents, column_headers_names)

        u_frame = pd.DataFrame(U[:, :5], index=column_headers)
        v_frame = pd.DataFrame(Vh[:5, :], columns=column_headers)
        u_frame.to_csv('u_1b_pca.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1b_pca.csv', index=True, encoding='utf-8')
        return (u_frame, v_frame, s)

if __name__ == "__main__":
    obj = PcaGenreActor()
    (u_frame, v_frame, s) = obj.pca_genre_actor(genre="Action")
    # print (u_frame)
    # print (v_frame)
    # print (s)
