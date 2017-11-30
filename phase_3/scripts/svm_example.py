import numpy
from sklearn import datasets
from utils import *
from svm import SupportVectorMachine
from collections import Counter
import operator

import config_parser
import data_extractor
from util import Util

util = Util()
conf = config_parser.ParseConfig()
data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
data_extractor_obj = data_extractor.DataExtractor(data_set_loc)
movie_tag_frame = util.get_movie_tag_matrix()

movie_tag_matrix_value = movie_tag_frame.values
(U, s, Vh) = util.SVD(movie_tag_matrix_value)
movie_latent_matrix = U[:, :5]

movies = list(movie_tag_frame.index.values)
tags = list(movie_tag_frame)
label_movies_json_data = data_extractor_obj.get_json()

def get_all_pairs_of_labels():
    labels = list(label_movies_json_data.keys())
    labels.sort()
    label_pairs = []
    for i in range(0, len(labels)-1):
        for j in range(i+1, len(labels)):
            label_pairs.append((labels[i],labels[j]))

    return label_pairs

def get_label_vectors_dict(label0, label1):
    indexes_of_label0_movies = [movies.index(each) for each in label_movies_json_data[label0]]
    indexes_of_label1_movies = [movies.index(each) for each in label_movies_json_data[label1]]
    matrix_of_label0_movies = movie_latent_matrix[indexes_of_label0_movies, :]
    matrix_of_label1_movies = movie_latent_matrix[indexes_of_label1_movies, :]

    label_vectors_dict = {}
    label_vectors_dict[-1] = matrix_of_label0_movies
    label_vectors_dict[1] = matrix_of_label1_movies

    return label_vectors_dict

def get_labelpair_clf_dict():
    label_pairs = get_all_pairs_of_labels()
    labelpair_clf_dict = {}
    for label0, label1 in label_pairs:
        label_vectors_dict = get_label_vectors_dict(label0, label1)
        concatenated_data = numpy.concatenate((label_vectors_dict[-1], label_vectors_dict[1]), axis=0)
        concatenated_labels = []
        for i in range(0, len(label_vectors_dict[-1])):
            concatenated_labels.append(-1)
        for j in range(0, len(label_vectors_dict[1])):
            concatenated_labels.append(1)
        concatenated_labels = numpy.array(concatenated_labels)
        clf = SupportVectorMachine(kernel=rbf_kernel, power=4, coef=1)
        clf.fit(concatenated_data, concatenated_labels)
        labelpair_clf_dict[(label0, label1)] = clf

        return labelpair_clf_dict

def get_label(labelpair_clf_dict, query_movie_id):
    movie_vector = movie_latent_matrix[movies.index(query_movie_id)]
    label_count = Counter()
    for label0, label1 in labelpair_clf_dict.keys():
        clf = labelpair_clf_dict[(label0, label1)]
        label_sign = clf.predict([movie_vector])
        if label_sign[0] == -1:
            label_count[label0] += 1
        else:
            label_count[label1] += 1
    label_count_dict_sorted = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    (label, max) = label_count_dict_sorted[0]

    return label


def starting_function():
    labelpair_clf_dict = get_labelpair_clf_dict()
    while True:
        query_movie_id = input("Enter the movie ID you want to know the predicted label: ")
        query_movie_id = int(query_movie_id)
        if query_movie_id not in movies:
            print("Invalid movie ID entered, hence skipping this movie!")
            continue
        predicted_label = get_label(labelpair_clf_dict, query_movie_id)
        print("Entered movie: " + str(query_movie_id) + " - " + util.get_movie_name_for_id(query_movie_id))
        print("Predicted label: " + str(predicted_label))
        confirmation = input("Are you done querying? (y/Y/n/N): ")
        if confirmation == "y" or confirmation == "Y":
            break