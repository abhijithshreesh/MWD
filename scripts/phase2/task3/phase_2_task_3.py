import logging
import operator
import numpy
import pandas as pd

from scripts.phase2.common.actor_actor_similarity_matrix import ActorActorMatrix
from scripts.phase2.common.coactor_coactor_matrix import CoactorCoactorMatrix
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
        self.actor_matrix, self.actorids = self.fetchActorActorSimilarityMatrix()
        self.coactor_obj = CoactorCoactorMatrix()
        self.coactor_matrix, self.coactorids = self.coactor_obj.fetchCoactorCoactorSimilarityMatrix()

    def get_transition_dataframe(self, data_frame):
        for column in data_frame:
            data_frame[column] = pd.Series(
                [0 if ind == int(column) else each for ind, each in zip(data_frame.index, data_frame[column])],
                index=data_frame.index)
        data_frame["row_sum"] = data_frame.sum(axis=1)
        for column in data_frame:
            data_frame[column] = pd.Series(
                [each / sum if (column != "row_sum" and each > 0 and ind != int(column)) else each for ind, each, sum in
                 zip(data_frame.index, data_frame[column], data_frame.row_sum)],
                index=data_frame.index)
        data_frame = data_frame.drop(["row_sum"], axis=1)
        data_frame = data_frame.transpose()
        return data_frame

    def compute_pagerank(self, seed_actors, actor_matrix, actorids):
        data_frame = pd.DataFrame(actor_matrix)
        transition_df = self.get_transition_dataframe(data_frame)
        seed_list = [0.0 for each in range(len(transition_df.columns))]
        seed_value = float(1/len(seed_actors))
        for each in seed_actors:
            seed_list[list(actorids).index(each)] = seed_value
        result_list = seed_list
        temp_list = []
        while(temp_list!=result_list):
            temp_list = result_list
            result_list = list(0.85*numpy.matmul(numpy.array(transition_df.values), numpy.array(result_list))+ 0.15*numpy.array(seed_list))
        page_rank_dict = {i: j for i, j in zip(actorids, result_list)}
        sorted_rank = sorted(page_rank_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_rank)

    def compute_actors_pagerank(self, seed_list):
        self.compute_pagerank(seed_list, self.actor_matrix, self.actorids)

    def compute_coactors_pagerank(self, seed_list):
        self.compute_pagerank(seed_list, self.coactor_matrix, self.coactorids)

if __name__ == "__main__":
    PRA = PageRankActor()
    PRA.compute_actors_pagerank([2055016])#2055016])
