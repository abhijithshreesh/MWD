import numpy

from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix


class ActorActorSVD(object):
    def actor_actor_svd(self):
        actor_actor_matrix_object = ActorActorMatrix()
        actor_actor_similarity_matrix, actor_ids = actor_actor_matrix_object.fetchActorActorSimilarityMatrix()
        U, s, V = numpy.linalg.svd(actor_actor_similarity_matrix, full_matrices=False)
        for list in V:
            print("\n\n\n")
            print(list)


if __name__ == "__main__":
    obj = ActorActorSVD()
    obj.actor_actor_svd()
