import tensorly.tensorly.decomposition as decomp
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_1 import ActorTag
import pandas as pd
import numpy as np
conf = ParseConfig()

class ActorMovieYearTensor(ActorTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def fetchActorMovieYearTensor(self):
        movies_df = self.data_extractor.get_mlmovies_data()
        actor_df = self.data_extractor.get_movie_actor_data()

        #Create Tensor(either list or dictionary)
        movie_actor_df = actor_df.merge(movies_df, how="left", on="movieid")
        tensor = np.ndarray(order=3)


        # tensor_dict = {}
        # for index, row in movie_actor_df.iterrows():
        #     year = row["year"]
        #     movieid = row["movieid"]
        #     actorid = row["actorid"]
        #     if year in tensor_dict:
        #         if movieid in tensor_dict[year]:
        #             tensor_dict[year][movieid].append(actorid)
        #         else:
        #             tensor_dict[year][movieid] = [actorid]
        #     else:
        #         tensor_dict[year] = {movieid:[actorid]}
        # return tensor_dict

        return tensor

    def cpDecomposition(self, tensor):
        #Add Tensorly package and use parafac()
        # a = np.array(pd.DataFrame.from_dict(tensor_dict))

        factors = decomp.parafac(tensor, 5)
        return factors

if __name__ == "__main__":
    obj = ActorMovieYearTensor()
    tensor = obj.fetchActorMovieYearTensor()
    obj.cpDecomposition(tensor)