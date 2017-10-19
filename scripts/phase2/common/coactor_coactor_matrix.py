import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_1 import ActorTag

conf = ParseConfig()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class CoactorCoactorMatrix(ActorTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def fetchCoactorCoactorSimilarityMatrix(self):
        movie_actor_df = self.data_extractor.get_movie_actor_data()
        movie_actor_set_df = movie_actor_df.groupby(['actorid'])["movieid"].apply(set).reset_index()
        num_of_actors = movie_actor_df.actorid.unique()
        coactor_matrix = [[0] * len(num_of_actors) for i in range(num_of_actors)]
        for index, movie_set in zip(movie_actor_set_df.index, movie_actor_set_df.movieid):
            for index_2, movie_set_2 in zip(movie_actor_set_df.index, movie_actor_set_df.movieid):
                if index != index_2:
                    coactor_matrix[index][index_2] = len(movie_set.intersection(movie_set_2))
        return coactor_matrix

if __name__ == "__main__":
    obj = CoactorCoactorMatrix()
    obj.fetchCoactorCoactorSimilarityMatrix()
