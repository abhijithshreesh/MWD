import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.util import Util
import numpy
from sklearn.decomposition import PCA
import logging
import pandas as pd
import operator
from scipy import linalg
from sklearn.preprocessing import StandardScaler
import numpy
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
from scripts.phase2.task1.phase_2_task_1b_svd import SvdGenreActor

log = logging.getLogger(__name__)
conf = ParseConfig()
util = Util()

class RelatedActorsPca():
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def get_related_actors_pca(self, actorid):
        """
        Triggers the compute function and outputs the result tag vector
        :param genre:
        :param model:
        :return: returns a dictionary of Genres to dictionary of tags and weights.
        """

        # Loading the required dataset
        df = pd.DataFrame(pd.read_csv('actor_tag_matrix.csv'))
        df1 = df.values

        # # Feature Scaling
        # sc = StandardScaler()
        # df_sc = sc.fit_transform(df1[:, :])
        #
        # # Computng covariance matrix
        # cov_df = numpy.cov(df_sc, rowvar=False)
        #
        # # Calculating PCA
        # U, s, Vh = linalg.svd(cov_df)

        (U, s, Vh) = util.PCA(df1)

        u_frame = pd.DataFrame(U[:, :5])
        v_frame = pd.DataFrame(Vh[:5, :])
        #u_frame.to_csv('u_1a_svd.csv', index=True, encoding='utf-8')
        #v_frame.to_csv('vh_1a_svd.csv', index=True, encoding='utf-8')
        #return (u_frame, v_frame, s)

        tag_latent_matrix = U[:, :5]
        actor_tag_matrix = df1
        actor_latent_matrix = numpy.dot(actor_tag_matrix, tag_latent_matrix)
        actorids = util.get_sorted_actor_ids()

        latent_actor_matrix = actor_latent_matrix.transpose()
        actor_actor_matrix = numpy.dot(actor_latent_matrix, latent_actor_matrix)
        numpy.savetxt("actor_actor_matrix_with_pca_latent_values.csv", actor_actor_matrix, delimiter=",")

        df = pd.DataFrame(pd.read_csv('actor_actor_matrix_with_svd_latent_values.csv', header=None))
        matrix = df.values

        actorids = util.get_sorted_actor_ids()

        index_actor = None
        for i, j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor == None:
            print("Actor Id not found.")
            return None

        actor_names = []
        for actor_id in actorids:
            actor_name = util.get_actor_name_for_id(int(actor_id))
            actor_names = actor_names + [actor_name]

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actor_names, actor_row))
        del actor_actor_dict[util.get_actor_name_for_id(int(actorid))]
        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        return actor_actor_dict

if __name__ == "__main__":
    obj = RelatedActorsPca()
    actor_actor_dict = obj.get_related_actors_pca(actorid=542238)
    print (actor_actor_dict)