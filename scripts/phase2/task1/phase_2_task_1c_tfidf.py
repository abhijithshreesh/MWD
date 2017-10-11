import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from collections import Counter
import math
import argparse
import operator
from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()

class SimilarActors(ActorActorMatrix):

    def __init__(self):
        """
        Initialiazing the data extractor object to get data from the csv files
        """
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def get_actor_actor_vector(self, actorid):
        (matrix, actorids) = self.fetchActorActorSimilarityMatrix()
        index_actor = None
        for i,j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor==None:
            print ("Actor Id not found.")
            return None

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actorids, actor_row))
        del actor_actor_dict[actorid]
        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        return actor_actor_dict


if __name__ == "__main__":
    obj = SimilarActors()
    actor_actor_dict = obj.get_actor_actor_vector(actorid=542238)
    print (actor_actor_dict)