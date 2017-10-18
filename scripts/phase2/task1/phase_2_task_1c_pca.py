import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.util import Util
import numpy
from sklearn.decomposition import PCA
import logging
import pandas as pd
import operator

import numpy
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
from scripts.phase2.task1.phase_2_task_1b_svd import SvdGenreActor

log = logging.getLogger(__name__)
conf = ParseConfig()
util = Util()

class RelatedActorsPCA(SvdGenreActor):
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
        #print(df1)
        # pca = PCA(n_components = 4)
        # pca.fit(x)
        # print(pca.explained_variance_ratio_)
        # print(pca.components_)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        df_sc = sc.fit_transform(df1[:, 1:])

        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        df_pca = pca.fit_transform(df_sc)
        numpy.savetxt("pca_actor_tag_for_related_actors.csv", df_pca, delimiter=",")
        explained_variance = pca.explained_variance_ratio_

        actor_latent_matrix = df_pca
        actorids = util.get_sorted_actor_ids()

        latent_actor_matrix = actor_latent_matrix.transpose()
        actor_actor_matrix = numpy.dot(actor_latent_matrix, latent_actor_matrix)
        numpy.savetxt("actor_actor_matrix_with_pca_latent_values.csv", actor_actor_matrix, delimiter=",")

        df = pd.DataFrame(pd.read_csv('actor_actor_matrix_with_pca_latent_values.csv', header=None))
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

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actorids, actor_row))
        del actor_actor_dict[actorid]
        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        return actor_actor_dict

if __name__ == "__main__":
    obj = RelatedActorsPCA()
    actor_actor_dict = obj.get_related_actors_pca(actorid=542238)
    print (actor_actor_dict)
