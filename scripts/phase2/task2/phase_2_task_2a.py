import numpy

from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
from scripts.phase2.common.util import Util


class ActorActorSVD(object):
    def __init__(self):
        self.actor_actor_matrix_object = ActorActorMatrix()
        self.actor_actor_similarity_matrix, self.actor_ids = self.actor_actor_matrix_object.fetchActorActorSimilarityMatrix()
        self.u, self.s, self.v = numpy.linalg.svd(self.actor_actor_similarity_matrix, full_matrices=False)
        self.vt = self.v.transpose()
        self.util = Util()

    def get_latent_semantics(self, r):
        latent_semantics = []
        for latent_semantic in self.vt:
            if len(latent_semantics) == r:
                break
            latent_semantics.append(latent_semantic)

        return latent_semantics

    def get_partitions(self, no_of_partitions):
        actor_names_list = []
        for actor in self.actor_ids:
            actor_names_list.append(self.util.get_actor_name_for_id(actor))
        groupings = self.util.partition_factor_matrix(self.u, no_of_partitions, actor_names_list)

        return groupings

    def print_partitioned_actors(self, no_of_partitions):
        groupings = self.get_partitions(no_of_partitions)
        for key in groupings.keys():
            print(str(key) + " Actors")
            for actor in groupings[key]:
                print(actor, end="|")
            print("\n")

    def print_latent_semantics(self, r):
        latent_semantics = self.get_latent_semantics(r)
        actor_names_list = []
        for actor in self.actor_ids:
            actor_names_list.append(self.util.get_actor_name_for_id(actor))

        for latent_semantic in latent_semantics:
            print("Latent Semantic in terms of Actors:")
            for i in range(0, len(actor_names_list)):
                print(str(latent_semantic[i]) + "*(" + str(actor_names_list[i]) + ")", end="")
                if i != len(actor_names_list) - 1:
                    print(" + ", end="")
            print("\n")


if __name__ == "__main__":
    obj = ActorActorSVD()
    obj.print_latent_semantics(3)
    print("\n\n\n")
    obj.print_partitioned_actors(3)
