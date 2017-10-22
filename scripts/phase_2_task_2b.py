import numpy

from coactor_coactor_matrix import CoactorCoactorMatrix
from util import Util


class CoactorCoactorSVD(object):
    def __init__(self):
        self.coactor_coactor_matrix_object = CoactorCoactorMatrix()
        self.coactor_coactor_similarity_matrix, self.actor_ids = self.coactor_coactor_matrix_object.fetchCoactorCoactorSimilarityMatrix()
        self.u, self.s, self.vt = numpy.linalg.svd(self.coactor_coactor_similarity_matrix, full_matrices=False)
        self.util = Util()

    def get_actor_names_list(self):
        actor_names_list = []
        for actor in self.actor_ids:
            actor_names_list.append(self.util.get_actor_name_for_id(actor))

        return actor_names_list

    def get_partitions(self, no_of_partitions):
        actor_names_list = self.get_actor_names_list()
        groupings = self.util.partition_factor_matrix(self.u, no_of_partitions, actor_names_list)

        return groupings

    def print_partitioned_actors(self, no_of_partitions):
        groupings = self.get_partitions(no_of_partitions)
        self.util.print_partitioned_entities(groupings)

    def print_latent_semantics(self, r):
        latent_semantics = self.util.get_latent_semantics(r, self.vt)
        actor_names_list = self.get_actor_names_list()
        self.util.print_latent_semantics(latent_semantics, actor_names_list)


if __name__ == "__main__":
    obj = CoactorCoactorSVD()
    obj.print_latent_semantics(3)
    print("\n\n\n")
    obj.print_partitioned_actors(3)
