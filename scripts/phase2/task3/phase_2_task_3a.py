import logging
import scipy
import networkx as nx
import operator
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
        actor_df["row_sum"] = actor_df.sum(axis=1)
        for column in actor_df:
            actor_df[column] = pd.Series(
                [float(each/sum) if (column != "row_sum" and each > 0 and ind != int(column)) else 0 for ind, each, sum in zip(actor_df.index, actor_df[column], actor_df.row_sum)],
                index=actor_df.index)
        actor_df = actor_df.drop(["row_sum"], axis=1)
        no_of_actors = len(actor_df.columns)
        actor_df.loc[(actor_df.T == 0).all()] = float(1 / (no_of_actors))
        seed_frame = pd.DataFrame(0.0, index=np.arange(no_of_actors), columns=actor_df.columns)
        seed_value = float(1/len(seed_actors))
        for each in seed_actors:
            seed_frame[list(actorids).index(each)] = seed_value
        seed_matrix = seed_frame.values
        res = 0.85*actor_df.values + 0.15*seed_matrix
        e_values, e_vectors = scipy.sparse.linalg.eigsh(res, k=1, sigma=1)
        page_rank_dict = {i: j[0] for i, j in zip(actorids, e_vectors)}
        sorted_rank = sorted(page_rank_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_rank)

if __name__ == "__main__":
    PRA = PageRankActor()
    PRA.get_similarity_matrix([2055016])#3619702, 3426176])#2055016])
