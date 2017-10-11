import logging

import numpy
from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
from scripts.phase2.common.config_parser import ParseConfig
from sklearn.decomposition import TruncatedSVD

from scripts.phase2.common.task_1 import ActorTag

log = logging.getLogger(__name__)
conf = ParseConfig()


class ActorActorSVD(ActorTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def actor_actor_svd(self):
        actor_actor_matrix_object = ActorActorMatrix()
        actor_actor_similarity_matrix = actor_actor_matrix_object.fetchActorActorSimilarityMatrix()
        U, s, Vh = numpy.linalg.svd(actor_actor_similarity_matrix.values, full_matrices=True)

        x = actor_actor_similarity_matrix.values
        svd = TruncatedSVD(n_components=3)
        svd.fit(x)
        print(svd.components_)


if __name__ == "__main__":
    obj = ActorActorSVD()
    obj.actor_actor_svd()
