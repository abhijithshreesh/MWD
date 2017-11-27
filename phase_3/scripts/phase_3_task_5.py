import random
import operator
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
movies = list(movie_tag_frame.index.values)
tags = list(movie_tag_frame)
label_movies_json_data = data_extractor_obj.get_json()

class ClassifierTask(object):
    def __init__(self, r=0):
        self.util = util
        self.r = r
        self.movie_tag_frame = movie_tag_frame
        self.movies = movies
        self.label_movies_json_data = label_movies_json_data
        self.movie_label_dict = self.get_labelled_movies()

    def get_labelled_movies(self):
        movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        movie_label_dict = {}
        print("Extracting labelled movies from the json provided!")
        for label in self.label_movies_json_data.keys():
            for movieid in self.label_movies_json_data[label]:
                if int(movieid) not in movie_id_list:
                    print("Invalid movie ID \'" + str(movieid) + "\' entered, hence skipping this movie!")
                    continue
                movie_name = self.util.get_movie_name_for_id(int(movieid))
                movie_label_dict[movie_name] = label
        print("Finished extracting!")

        return movie_label_dict

    def find_label_RNN(self, query_movie_name):
        movie_tag_matrix = self.movie_tag_frame.values
        query_movie_index = self.movies.index(query_movie_name)
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
        movie_value_for_tag = movie_tag_frame.values[movies.index(movie)][tree.feature_index]
        if movie_value_for_tag > 0:
            return self.predict_using_DTC(tree.right, movie)
        else:
            return self.predict_using_DTC(tree.left, movie)

    def demo_output(self, model):
        movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        predicted_label = None
        tree = None
        if model == "DTC":
            node = Node(self.movie_label_dict)
            tree = node.construct_tree()
        while True:
            query_movie_id = input("Enter the movie ID you want to know the predicted label: ")
            query_movie_id = int(query_movie_id)
            if query_movie_id not in movie_id_list:
                print("Invalid movie ID entered, hence skipping this movie!")
                continue
            query_movie_name = self.util.get_movie_name_for_id(query_movie_id)
            if model == "RNN":
                predicted_label = self.find_label_RNN(query_movie_name)
            elif model == "DTC":
                predicted_label = self.predict_using_DTC(tree, query_movie_name)
            print("Entered movie: " + query_movie_name + "\nPredicted label: " + str(predicted_label))
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
        self.feature_index = random.randint(0, len(tags) - 1)

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
            movie_value_for_tag = movie_tag_frame.values[movies.index(movie)][self.feature_index]
            if movie_value_for_tag > 0:
                right_movie_label_dict[movie] = label
            else:
                left_movie_label_dict[movie] = label
        self.left = Node(left_movie_label_dict).construct_tree()
        self.right = Node(right_movie_label_dict).construct_tree()

        return self


if __name__ == "__main__":
    r = 4
    model = "DTC" #RNN,SVM,DTC
    obj = ClassifierTask(r)
    obj.demo_output(model)