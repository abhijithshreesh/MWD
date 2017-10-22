import math

import numpy
import tensorly.tensorly.decomposition as decomp
from scipy import linalg
from sklearn.preprocessing import StandardScaler

from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor
from gensim import corpora, models
import gensim


class Util(object):

    """
    Class to relate actors and tags.
    """

    def __init__(self):
        """
        Initializing the data extractor object to get data from the csv files
        """
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.mlratings = self.data_extractor.get_mlratings_data()
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.imdb_actor_info = self.data_extractor.get_imdb_actor_info_data()
        self.genome_tags = self.data_extractor.get_genome_tags_data()

    def get_sorted_actor_ids(self):
        actor_info = self.data_extractor.get_imdb_actor_info_data()
        actorids = actor_info.id
        actorids = actorids.sort_values()
        return actorids

    def get_movie_id(self, movie):
        all_movie_data = self.mlmovies
        movie_data = all_movie_data[all_movie_data['moviename'] == movie]
        movie_id = movie_data['movieid'].unique()

        return movie_id[0]

    def get_average_ratings_for_movie(self, movie_id):
        all_ratings = self.mlratings
        movie_ratings = all_ratings[all_ratings['movieid'] == movie_id]

        ratings_sum = 0
        ratings_count = 0
        for index, row in movie_ratings.iterrows():
            ratings_count += 1
            ratings_sum += row['rating']

        return ratings_sum / float(ratings_count)

    def get_actor_name_for_id(self, actor_id):
        actor_data = self.imdb_actor_info[self.imdb_actor_info['id'] == actor_id]
        name = actor_data['name'].unique()

        return name[0]

    def get_movie_name_for_id(self, movieid):
        all_movie_data = self.mlmovies
        movie_data = all_movie_data[all_movie_data['movieid'] == movieid]
        movie_name = movie_data['moviename'].unique()

        return movie_name[0]

    def get_tag_name_for_id(self, tag_id):
        tag_data = self.genome_tags[self.genome_tags['tagId'] == tag_id]
        name = tag_data['tag'].unique()

        return name[0]

    def partition_factor_matrix(self, matrix, no_of_partitions, entity_names):
        entity_dict = {}
        for i in range(0, len(matrix)):
            length = 0
            for latent_semantic in matrix[i]:
                length += abs(latent_semantic)
            entity_dict[entity_names[i]] = length

        max_length = max(entity_dict.values())
        min_length = min(entity_dict.values())
        length_of_group = (float(max_length) - float(min_length)) / float(no_of_partitions)

        groups = {}
        for i in range(0, no_of_partitions):
            groups["Group " + str(i + 1)] = []

        for key in entity_dict.keys():
            entity_length = entity_dict[key]
            group_no = math.ceil(float(entity_length - min_length) / float(length_of_group))
            if group_no == 0:
                group_no = 1
            groups["Group " + str(group_no)].append(key)

        return groups

    def get_latent_semantics(self, r, matrix):
        latent_semantics = []
        for latent_semantic in matrix:
            if len(latent_semantics) == r:
                break
            latent_semantics.append(latent_semantic)

        return latent_semantics

    def print_partitioned_entities(self, groupings):
        for key in groupings.keys():
            print(key)
            if len(groupings[key]) == 0:
                print("NO ELEMENTS IN THIS GROUP\n")
                continue
            for entity in groupings[key]:
                print(entity, end="|")
            print("\n")

    def print_latent_semantics(self, latent_semantics, entity_names_list):
        for latent_semantic in latent_semantics:
            print("Latent Semantic:")
            for i in range(0, len(entity_names_list)):
                print(str(latent_semantic[i]) + "*(" + str(entity_names_list[i]) + ")", end="")
                if i != len(entity_names_list) - 1:
                    print(" + ", end="")
            print("\n")

    def CPDecomposition(self, tensor, rank):
        factors = decomp.parafac(tensor, rank)
        return factors

    def SVD(self, matrix):
        # Feature Scaling
        sc = StandardScaler()
        df_sc = sc.fit_transform(matrix[:, :])

        # Calculating SVD
        U, s, Vh = linalg.svd(df_sc)
        return (U, s, Vh)

    def PCA(self, matrix):
        # Feature Scaling
        sc = StandardScaler()
        df_sc = sc.fit_transform(matrix[:, :])

        # Computng covariance matrix
        cov_df = numpy.cov(df_sc, rowvar=False)

        # Calculating PCA
        U, s, Vh = linalg.svd(cov_df)
        return (U, s, Vh)

    def LDA(self, input_compound_list, num_topics, num_features):
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(input_compound_list)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in input_compound_list]

        # generate LDA model
        lda = gensim.models.ldamodel.LdaModel(corpus, num_topics, id2word=dictionary, passes=1)

        latent_semantics = lda.print_topics(num_topics, num_features)
        # for latent in latent_semantics:
        #     print(latent)

        corpus = lda[corpus]

        # for i in corpus:
        #     print(i)

        return (corpus, latent_semantics)

    def get_doc_topic_matrix(self, u, num_docs, num_topics):
        u_matrix = numpy.zeros(shape=(num_docs, num_topics))

        for i in range(0, len(u)):
            row1 = u[i]
            for j in range(0, len(row1)):
                u_matrix[i, j] = row1[j][1]

        # u_matrix = numpy.array(data)
        return u_matrix


if __name__ == "__main__":
    obj = Util()
    actorids = obj.get_sorted_actor_ids()
    print("Actorids Sorted along with original index: \n")
    print(actorids)
