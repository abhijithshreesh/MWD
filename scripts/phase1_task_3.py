import logging

import pandas as pd

from scripts.config_parser import ParseConfig
from scripts.phase1_task_2 import GenreTag

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class UserTag(GenreTag):
    """
          Class to relate Users and tags, inherits the GenreTag to use the common weighing functons
    """
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def combine_computed_weights(self, data_frame, model, user):
        """
                Triggers the weighing process and sums up all the calculated weights for each tag
                :param data_frame:
                :param rank_weight_dict:
                :param model:
                :return: dictionary of tags and weights
        """
        tag_df = data_frame.reset_index()
        temp_df = data_frame[data_frame["userid"] == user]
        unique_tags = tag_df.tag.unique()
        idf_data = tag_df.groupby(['movieid'])['tag'].apply(set)
        tf_df = temp_df.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag))
        tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in
                          list(movie_tag_dict.items())}
        idf_weight_dict = {}
        if model != 'TF':
            idf_weight_dict = self.assign_idf_weight(idf_data, unique_tags)
        tag_df = self.get_model_weight(tf_weight_dict, idf_weight_dict, temp_df, model)
        tag_df["total"] = tag_df.groupby(['tag'])['value'].transform('sum')
        tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
        # actor_tag_dict = dict(zip(tag_df.tag, tag_df.total))
        return tag_df


    def merge_genre_tag(self, user, model):
        """
        Merges data from different csv files necessary to compute the tag weights for each user,
        assigns weights to timestamp.
        :param user:
        :param model:
        :return: returns a dictionary of Users to dictionary of tags and weights.
        """
        genome_tag = self.data_extractor.get_genome_tags_data()
        ml_tag = self.data_extractor.get_mltags_data()
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        data_frame_len = len(tag_data_frame.index)
        tag_data_frame["timestamp_weight"] = pd.Series(
            [(index + 1) / data_frame_len * 10 for index in tag_data_frame.index],
            index=tag_data_frame.index)

        tag_dict = self.combine_computed_weights(tag_data_frame, model, user)
        print({user: tag_dict})

if __name__ == "__main__":
    obj = UserTag()
    # parser = argparse.ArgumentParser(description='task3.py userid model')
    # parser.add_argument('userid', action="store", type=int)
    # parser.add_argument('model', action="store", type=str, choices=set(('TF', 'TFIDF')))
    # input = vars(parser.parse_args())
    # userid = input['userid']
    # model = input['model']
    # obj.merge_genre_tag(user=userid, model=model)
    obj.merge_genre_tag(user=109, model="TFIDF")

