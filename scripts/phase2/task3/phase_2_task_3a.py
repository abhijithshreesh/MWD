import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from collections import Counter
import math
import argparse
import operator
from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
import networkx as nx
import numpy as np
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
            actor_df[column] = pd.Series([int(column) if each>0 else 0 for each in actor_df[column]], index= actor_df.index)
        graph = nx.from_numpy_matrix(np.array(actor_df.values))
        pagerank_dict = nx.pagerank(graph)
        a =1


if __name__ == "__main__":
    PRA = PageRankActor()
    PRA.get_similarity_matrix()
