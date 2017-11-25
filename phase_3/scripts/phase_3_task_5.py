import numpy
import operator
from collections import Counter

import config_parser
import data_extractor

from util import Util

class ClassifierTask(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.util = Util()
        self.movie_tag_frame = self.util.get_movie_tag_matrix()
        self.movies = list(self.movie_tag_frame.index.values)

    def get_labelled_movies(self):
        movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        movie_label_dict = {}
        print("Provide set of labelled movies")
        while True:
            label = input("Enter the label: ")
            label = str(label)
            movieids = input("\nPlease enter comma separated ids of the movies belonging to label \'" + label + "\': ")
            movieids = set(movieids.strip(" ").strip(",").replace(" ", "").split(","))
            for movieid in movieids:
                if int(movieid) not in movie_id_list:
                    print("Invalid movie ID \'"+ movieid +"\' entered, hence skipping this movie!")
                    continue
                movie_name = self.util.get_movie_name_for_id(int(movieid))
                movie_label_dict[movie_name] = label
            confirmation = input("Are you done entering labelled movies? (y/Y/n/N): ")
            if confirmation == "y" or confirmation == "Y":
                break

        return movie_label_dict

    def get_movie_movie_distance_matrix(self):
        movie_tag_matrix = self.movie_tag_frame.values
        tag_movie_matrix = movie_tag_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_tag_matrix, tag_movie_matrix)
        return movie_movie_matrix

    def predict_label(self, movie, distance_to_labelled_movies, movie_label_dict, r):
        label_count_dict = Counter()
        distance_to_labelled_movies_sorted = sorted(distance_to_labelled_movies.items(), key=operator.itemgetter(1))
        distance_to_labelled_movies_sorted = distance_to_labelled_movies_sorted[0:r]
        for (labelled_movie, distance) in distance_to_labelled_movies_sorted:
            label_count_dict[movie_label_dict[labelled_movie]] += 1
        label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        return label_count_dict_sorted[0]

    def predict_label_for_all_movies(self):
        movie_label_dict = self.get_labelled_movies()
        print("Classifying the rest of the movies, please wait!")
        movie_movie_matrix = self.get_movie_movie_distance_matrix()
        distance_to_labelled_movies = {}
        predicted_movie_label_dict = {}
        r = input("Enter the value of r: ")
        r = int(r)
        for i in range(0, len(self.movies)):
            labelled_movies = movie_label_dict.keys()
            if self.movies[i] in labelled_movies:
                continue
            for labelled_movie in labelled_movies:
                distance_to_labelled_movies[labelled_movie] = movie_movie_matrix[i][self.movies.index(labelled_movie)]
            (label, count) = self.predict_label(self.movies[i], distance_to_labelled_movies, movie_label_dict, r)
            predicted_movie_label_dict[self.movies[i]] = label

        return predicted_movie_label_dict

    def demo_output(self):
        movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        predicted_movie_label_dict = self.predict_label_for_all_movies()
        print("Finished classifying!")
        while True:
            query_movie_id = input("Enter the movie ID you want to know the predicted label: ")
            query_movie_id = int(query_movie_id)
            if query_movie_id not in movie_id_list:
                print("Invalid movie ID entered, hence skipping this movie!")
                continue
            query_movie_name = self.util.get_movie_name_for_id(query_movie_id)
            predicted_label = predicted_movie_label_dict[query_movie_name]
            print("Entered movie: " + query_movie_name + "\nPredicted label: " + predicted_label)
            confirmation = input("Are you done querying? (y/Y/n/N): ")
            if confirmation == "y" or confirmation == "Y":
                break


if __name__ == "__main__":
    obj = ClassifierTask()
    obj.demo_output()