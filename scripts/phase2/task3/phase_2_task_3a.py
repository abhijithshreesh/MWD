import logging
import scipy
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

    def get_similarity_matrix(self, seed_actors):
        actor_matrix, actorids = self.fetchActorActorSimilarityMatrix()
        actor_df = pd.DataFrame(actor_matrix)
        for column in actor_df:
            actor_df[column] = pd.Series(
                [1 if (each > 0 and ind != int(column)) else 0 for ind, each in zip(actor_df.index, actor_df[column])],
                index=actor_df.index)
        seed_frame = pd.DataFrame(0 ,index = np.arange(len(actor_df.columns)), columns=actor_df.columns)
        seed_matrix = seed_frame.values
        for each in seed_actors:
            seed_matrix[list(actorids).index(each), list(actorids).index(each)] = 1/len(seed_actors)
        actor_df.loc[(actor_df.T == 0).all()] = 1 / (len(actor_df.columns))
        res = 0.85*actor_df.values + 0.15*seed_matrix
        e_values, e_vectors = scipy.sparse.linalg.eigsh(res, k=1, sigma=1)
        page_rank_dict = {i:round(float(j[0]), 10) for i,j in zip(actorids, e_vectors)}

        # personalization = {each: 0 for each in actorids.index}
        # personalization[list(actorids).index(581911)] = 1
        # graph = nx.from_numpy_matrix(np.array(actor_df.values))
        # pagerank_dict = nx.pagerank(graph, personalization=personalization)
        a =1


if __name__ == "__main__":
    PRA = PageRankActor()
    PRA.get_similarity_matrix([581911, 2413133, 17838])
