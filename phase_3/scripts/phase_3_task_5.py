import operator
import numpy
from collections import Counter
from scipy.spatial import distance as dist
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
movie_latent_matrix = U[:, :10]

movies = list(movie_tag_frame.index.values)
tags = list(movie_tag_frame)
label_movies_json_data = data_extractor_obj.get_json()

class ClassifierTask(object):
    def __init__(self, r=0):
        self.util = util
        self.r = r
        self.movie_latent_matrix = movie_latent_matrix
        self.movies = movies
        self.label_movies_json_data = label_movies_json_data
        self.movie_label_dict = self.get_labelled_movies()

    def get_labelled_movies(self):
        #movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        movie_label_dict = {}
        print("Extracting labelled movies from the json provided!")
        for label in self.label_movies_json_data.keys():
            for movieid in self.label_movies_json_data[label]:
                if int(movieid) not in self.movies:
                    print("Invalid movie ID \'" + str(movieid) + "\' entered, hence skipping this movie!")
                    continue
                movie_name = self.util.get_movie_name_for_id(int(movieid))
                movie_label_dict[int(movieid)] = label
        print("Finished extracting!")

        return movie_label_dict

    def find_label_RNN(self, query_movie_id):
        movie_tag_matrix = self.movie_latent_matrix
        query_movie_index = self.movies.index(query_movie_id)
        query_movie_vector = movie_tag_matrix[query_movie_index]
        distance_to_labelled_movies = {}
        for labelled_movie in self.movie_label_dict.keys():
            labelled_movie_index = self.movies.index(labelled_movie)
            labelled_movie_vector = movie_tag_matrix[labelled_movie_index]
            distance_to_labelled_movies[labelled_movie] = dist.euclidean(query_movie_vector, labelled_movie_vector)
        label_count_dict = Counter()
        distance_to_labelled_movies_sorted = sorted(distance_to_labelled_movies.items(), key=operator.itemgetter(1))
        distance_to_labelled_movies_sorted = distance_to_labelled_movies_sorted[0:self.r]
        for (labelled_movie, distance) in distance_to_labelled_movies_sorted:
            label_count_dict[self.movie_label_dict[labelled_movie]] += 1
        label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        (label, count) = label_count_dict_sorted[0]

        return label

    def predict_using_DTC(self, tree, movie):
        if tree.dominant_label != False:
            return tree.dominant_label
        movie_value_for_tag = self.movie_latent_matrix[movies.index(movie)][tree.feature_index]
        if movie_value_for_tag > tree.mean_value:
            return self.predict_using_DTC(tree.right, movie)
        else:
            return self.predict_using_DTC(tree.left, movie)

    def demo_output(self, model):
        #movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        predicted_label = None
        tree = None
        if model == "DTC":
            node = Node(self.movie_label_dict)
            tree = node.construct_tree()
        while True:
            query_movie_id = input("Enter the movie ID you want to know the predicted label: ")
            query_movie_id = int(query_movie_id)
            if query_movie_id not in self.movies:
                print("Invalid movie ID entered, hence skipping this movie!")
                continue
            if model == "RNN":
                predicted_label = self.find_label_RNN(query_movie_id)
            elif model == "DTC":
                predicted_label = self.predict_using_DTC(tree, query_movie_id)
            print("Entered movie: " + str(query_movie_id) + " - " + self.util.get_movie_name_for_id(query_movie_id))
            print("Predicted label: " + str(predicted_label))
            confirmation = input("Are you done querying? (y/Y/n/N): ")
            if confirmation == "y" or confirmation == "Y":
                break

class Node(object):
    def __init__(self, movie_label_dict):
        self.left = None
        self.right = None
        self.data = movie_label_dict
        self.dominant_label = False
        self.parent = None
        self.feature_index = self.find_best_feature()
        self.mean_value = self.find_mean()

    def find_mean(self):
        if self.feature_index == None:
            return None
        indexes_of_movies = [movies.index(each) for each in self.data.keys()]
        matrix_of_movies = movie_latent_matrix[indexes_of_movies, self.feature_index]
        mean_of_movies = numpy.nanmean(matrix_of_movies)

        return mean_of_movies

    def calculate_fsr(self, feature_index, movies_of_class1, movies_of_class2):
        #a[[0, 1, 3], 2]
        indexes_of_class1_movies = [movies.index(each) for each in movies_of_class1]
        indexes_of_class2_movies = [movies.index(each) for each in movies_of_class2]
        matrix_of_class1_movies = movie_latent_matrix[indexes_of_class1_movies, feature_index]
        matrix_of_class2_movies = movie_latent_matrix[indexes_of_class2_movies, feature_index]
        mean_of_class1 = numpy.nanmean(matrix_of_class1_movies)
        mean_of_class2 = numpy.nanmean(matrix_of_class2_movies)
        variance_of_class1 = numpy.nanvar(matrix_of_class1_movies)
        variance_of_class2 = numpy.nanvar(matrix_of_class2_movies)

        #fsr(f) = (m1 - m2)^2 / (v1^2 + v2^2)
        fsr = ((mean_of_class1 - mean_of_class2)**2) / ((variance_of_class1**2) + (variance_of_class2**2))

        return fsr

    def find_best_feature(self):
        labels = list(set(self.data.values()))
        if len(labels) < 2:
            #Return None as its not gonna be further divided
            return None
        movieids = list(self.data.keys())
        label_movie_dict = {}
        for movieid in movieids:
            label = self.data[movieid]
            if label in label_movie_dict.keys():
                label_movie_dict[label].append(movieid)
            else:
                label_movie_dict[label] = [movieid]
        fsr_list = []
        for feature_index in range(0, 10):
            fsr = 0
            for i in range(0, len(labels) - 1):
                for j in range(i+1, len(labels)):
                    fsr += self.calculate_fsr(feature_index, label_movie_dict[labels[i]], label_movie_dict[labels[j]])
            avg_fsr = fsr / (len(labels) * (len(labels) - 1) / 2)
            fsr_list.append(avg_fsr)
        best_feature_index, value = max(enumerate(fsr_list), key=operator.itemgetter(1))

        return best_feature_index

    def check_dominancy(self, movie_label_dict_values):
        label_count_dict = Counter(movie_label_dict_values)
        label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        (dominant_label, dominant_count) = label_count_dict_sorted[0]
        dominancy = float(dominant_count) / len(movie_label_dict_values)
        if dominancy > 0.75:
            return dominant_label
        else:
            return False

    def construct_tree(self):
        movie_label_dict_values = self.data.values()
        if len(movie_label_dict_values) == 0:
            return None
        dominant_label = self.check_dominancy(movie_label_dict_values)
        if dominant_label:
            self.dominant_label = dominant_label
            return self
        left_movie_label_dict = {}
        right_movie_label_dict = {}
        for (movie, label) in self.data.items():
            movie_value_for_tag = movie_latent_matrix[movies.index(movie)][self.feature_index]
            if movie_value_for_tag > self.mean_value:
                right_movie_label_dict[movie] = label
            else:
                left_movie_label_dict[movie] = label
        self.left = Node(left_movie_label_dict).construct_tree()
        self.right = Node(right_movie_label_dict).construct_tree()

        return self


if __name__ == "__main__":
    r = 1
    model = "RNN" #RNN,SVM,DTC
    obj = ClassifierTask(r)
    obj.demo_output(model)