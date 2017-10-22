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
from scripts.phase2.common.data_extractor import DataExtractor

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()

class LdaActorTag(object):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def get_lda_data(self, genre):
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

        tag_df = list(tag_df.iloc[:,1])



        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(tag_df)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in tag_df]

        # generate LDA model
        lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=1)

        latent_semantics = lda.print_topics(5,80)
        for i in range(0, len(latent_semantics)):
            print (latent_semantics[i])

        #print (lda.print_topics(5,80))

        corpus = lda[corpus]

        for i in range(0, len(corpus)):
            if len(corpus[i]) == 1:
                print("\n\n\n")
                print(i)
                print("\n\n\n")
            print(corpus[i])



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
    obj = LdaActorTag()
    lda_comp = obj.get_lda_data(genre="Action")
    #print (lda_comp)