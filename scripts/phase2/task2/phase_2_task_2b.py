import logging

import numpy
from sklearn.decomposition import TruncatedSVD

from scripts.phase2.common.coactor_coactor_matrix import CoactorCoactorMatrix
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_1 import ActorTag

log = logging.getLogger(__name__)
conf = ParseConfig()

class CoactorCoactorSVD(ActorTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def coactor_coactor_svd(self):
        coactor_coactor_matrix_object = CoactorCoactorMatrix()
        coactor_coactor_similarity_matrix, actor_ids = coactor_coactor_matrix_object.fetchCoactorCoactorSimilarityMatrix()
        # print(coactor_coactor_similarity_matrix)
        U, s, Vh = numpy.linalg.svd(coactor_coactor_similarity_matrix, full_matrices=False)

        x = coactor_coactor_similarity_matrix
        svd = TruncatedSVD(n_components=3)
        svd.fit(x)
        print(svd.components_)

        # print(U)
        # print(s)
        # print(Vh)


if __name__ == "__main__":
    obj = CoactorCoactorSVD()
    obj.coactor_coactor_svd()
