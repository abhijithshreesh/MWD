import logging

import networkx as nx
import numpy as np
import pandas as pd

from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()

class PageRankActor(ActorActorMatrix):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def get_similarity_matrix(self):
        actor_matrix, actorids = self.fetchActorActorSimilarityMatrix()
        actor_df = pd.DataFrame(actor_matrix)
        for column in actor_df:
            actor_df[column] = pd.Series([1 if each > 0 else 0 for each in actor_df[column]], index=actor_df.index)
        temp_df = actor_df[(actor_df.T == 0).any()]
        actor_df.loc[(actor_df.T == 0).any()] = 1 / (len(temp_df.columns))
        graph = nx.from_numpy_matrix(np.array(actor_df.values))
        pagerank_dict = nx.pagerank(graph)
        a =1


if __name__ == "__main__":
    PRA = PageRankActor()
    PRA.get_similarity_matrix()
