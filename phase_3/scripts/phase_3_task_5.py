import operator
from collections import Counter
from scipy.spatial import distance as dist
import config_parser
import data_extractor
from util import Util

class ClassifierTask(object):
    def __init__(self, r=0):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.util = Util()
        self.r = r
        self.movie_tag_frame = self.util.get_movie_tag_matrix()
        self.movies = list(self.movie_tag_frame.index.values)
        self.label_movies_json_data = self.data_extractor.get_json()
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

    def demo_output(self, model):
        movie_id_list = [self.util.get_movie_id(each) for each in self.movies]
        predicted_label = None
        while True:
            query_movie_id = input("Enter the movie ID you want to know the predicted label: ")
            query_movie_id = int(query_movie_id)
            if query_movie_id not in movie_id_list:
                print("Invalid movie ID entered, hence skipping this movie!")
                continue
            query_movie_name = self.util.get_movie_name_for_id(query_movie_id)
            if model == "RNN":
                predicted_label = self.find_label_RNN(query_movie_name)
            print("Entered movie: " + query_movie_name + "\nPredicted label: " + str(predicted_label))
            confirmation = input("Are you done querying? (y/Y/n/N): ")
            if confirmation == "y" or confirmation == "Y":
                break


if __name__ == "__main__":
    r = 3
    model = "RNN" #RNN,SVM,DTC
    obj = ClassifierTask(r)
    obj.demo_output(model)