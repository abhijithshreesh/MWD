import config_parser
import data_extractor
import numpy
import pandas as pd
from phase1_task_2 import GenreTag
from util import Util

class ClassifierTask(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)

    def get_data_for_classifier(self):
        return None

if __name__ == "__main__":
    obj = ClassifierTask()
    data = obj.get_data_for_classifier()