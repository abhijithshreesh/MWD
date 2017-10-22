import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_2 import GenreTag
from sklearn.preprocessing import Imputer
from sklearn.decomposition import LatentDirichletAllocation
import argparse
from collections import Counter
from gensim import corpora, models
import gensim
import numpy
import operator
from scripts.phase2.common.data_extractor import DataExtractor
from scripts.phase2.common.util import Util

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class LdaActorTag(object):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.util = Util()

    def get_related_actors_lda(self, actorid):
        mov_act = self.data_extractor.get_movie_actor_data()
        ml_tag = self.data_extractor.get_mltags_data()
        genome_tag = self.data_extractor.get_genome_tags_data()
        actor_info = self.data_extractor.get_imdb_actor_info_data()
        actor_movie_info = mov_act.merge(actor_info, how="left", left_on="actorid", right_on="id")
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        merged_data_frame = tag_data_frame.merge(actor_movie_info, how="left", on="movieid")

        merged_data_frame = merged_data_frame.fillna('')
        tag_df = merged_data_frame.groupby(['actorid'])['tag'].apply(list).reset_index()

        tag_df = tag_df.sort_values('actorid')
        actorid_list = tag_df.actorid.tolist()
        tag_df = list(tag_df.iloc[:,1])

        (U, Vh) = self.util.LDA(tag_df, num_topics=5, num_features=1000)

        actor_topic_matrix = self.util.get_doc_topic_matrix(U, num_docs=len(actorid_list), num_topics=5)
        topic_actor_matrix = actor_topic_matrix.transpose()
        actor_actor_matrix = numpy.dot(actor_topic_matrix,topic_actor_matrix)

        numpy.savetxt("actor_actor_matrix_with_svd_latent_values.csv", actor_actor_matrix, delimiter=",")

        df = pd.DataFrame(pd.read_csv('actor_actor_matrix_with_svd_latent_values.csv', header=None))
        matrix = df.values

        actorids = self.util.get_sorted_actor_ids()

        index_actor = None
        for i, j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor == None:
            print("Actor Id not found.")
            return None

        actor_names = []
        for actor_id in actorids:
            actor_name = self.util.get_actor_name_for_id(int(actor_id))
            actor_names = actor_names + [actor_name]

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actor_names, actor_row))
        del actor_actor_dict[self.util.get_actor_name_for_id(int(actorid))]

        # for key in actor_actor_dict.keys():
        #     actor_actor_dict[key] = abs(actor_actor_dict[key])

        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        return actor_actor_dict


if __name__ == "__main__":
    obj = LdaActorTag()
    actor_actor_dict = obj.get_related_actors_lda(actorid=542238)
    print(actor_actor_dict)