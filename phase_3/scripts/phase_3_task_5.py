import config_parser
import data_extractor
from util import Util

class ClassifierTask(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.util = Util()

    def get_data_for_classifier(self):
        movie_tag_frame = self.util.get_movie_tag_matrix()
        return movie_tag_frame

if __name__ == "__main__":
    obj = ClassifierTask()
    data = obj.get_data_for_classifier()