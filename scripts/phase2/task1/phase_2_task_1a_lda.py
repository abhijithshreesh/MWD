import pandas as pd
import logging
from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.task_2 import GenreTag
from sklearn.preprocessing import Imputer
from sklearn.decomposition import LatentDirichletAllocation
import argparse
#import gensim
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class LdaGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def get_lda_data(self, genre):
         data_frame = self.get_genre_data().reset_index()
         genre_data_frame = data_frame[data_frame["genre"]==genre]
         tf_df = genre_data_frame.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
         movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag))
         tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in
                           list(movie_tag_dict.items())}
         tag_df = self.get_model_weight(tf_weight_dict, {}, genre_data_frame, "TF")
         tag_df["value"] = pd.Series([tf_weight_dict.get(movieid, 0).get(tag, 0) for index, tag, movieid
                             in zip(tag_df.index, tag_df.tag, tag_df.movieid)], index=tag_df.index)
         #tag_df["total"] = tag_df.groupby(['movieid', 'tag'])['value'].transform('sum')
         tag_df = tag_df.drop_duplicates("tag").sort_values("value", ascending=False)
         temp_df = tag_df[["moviename", "tag", "value"]].drop_duplicates()
         genre_tag_freq = temp_df.pivot(index='moviename', columns='tag', values='value') #, fill_value=0)
         genre_tag_freq = genre_tag_freq.fillna(0)
         lda = LatentDirichletAllocation(n_components=4)
         lda.fit(genre_tag_freq.values)
         lda_comp = lda.components_
         return lda_comp

        #lda = LatentDirichletAllocation(n_topics=4)

        #lda.fit_transform(genre_tag_freq.values)
        #topics = lda.components_

        #ldamodel = gensim.models.ldamodel.LdaModel(tag_df.values, num_topics=3)
        #print(ldamodel.print_topics(num_topics=3, num_words=3))

        # Loading the dataset
        #df = pd.DataFrame(pd.read_csv('tag_df_lda.csv'))
        #df1 = df.values
        #print(df1)

        # Encoding the String Variables

        # from sklearn.preprocessing import LabelEncoder
        # labelencoder_df = LabelEncoder()
        # df.iloc[:, 1] = labelencoder_df.fit_transform(df.iloc[:, 1])
        # df.iloc[:, 2] = labelencoder_df.fit_transform(df.iloc[:, 2])
        #
        # df1 = df.values

        # Calling the LDA algorithm


if __name__ == "__main__":
    obj = LdaGenreTag()
    lda_comp = obj.get_lda_data(genre="Action")
    print (lda_comp)