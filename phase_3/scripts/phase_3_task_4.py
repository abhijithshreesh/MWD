import pandas as pd
import data_extractor
import config_parser
from util import Util
from phase_3_task_3 import MovieLSH

class RelevancyFinder():

    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.util = Util()
        self.movie_tag_df = self.util.get_movie_tag_matrix()
        self.movie_tag_df = self.movie_tag_df.reset_index()
        self.relevancy_df = self.data_extractor.get_task4_feedback_data()


    def query_point(self, old_query_point):
        self.relevancy_df = self.data_extractor.get_task4_feedback_data()
        self.relevancy_df = self.relevancy_df.set_index('moviename')
        merged_data_frame = self.relevancy_df.reset_index().merge(self.movie_tag_df, how="left", on="moviename")
        merged_data_frame = merged_data_frame.set_index('moviename')
        merged_data_frame = merged_data_frame.sort_values('relevancy')
        query_point = merged_data_frame.groupby('relevancy', axis=0).mean()
        #query_point.to_csv(self.data_set_loc + '/temp1.csv', index=True, encoding='utf-8')
        query_point = query_point.reset_index()
        del query_point['relevancy']
        if query_point.shape[0] < 2:
            query_point = old_query_point + query_point.iloc[0]
        else:
            query_point = old_query_point + query_point.iloc[1] - query_point.iloc[0]
        #query_point.to_csv(self.data_set_loc + '/temp2.csv', index=True, encoding='utf-8')
        #merged_data_frame.to_csv(self.data_set_loc + '/temp.csv', index=True, encoding='utf-8')
        return query_point

    def relevancy(self, no_r,query_point):
        #movie_names = MovieLSH.query_for_nearest_neighbours(self, query_point, no_r)
        movie_names = self.movie_tag_df['moviename'].sample(n=no_r)
        Util.print_movie_recommendations_and_collect_feedback(self, movie_names, 4, None)


if __name__ == "__main__":
    obj = RelevancyFinder()
    i = 1
    qp = obj.query_point(0)
    while i:
        r = int(input("\nEnter value of 'r' : "))
        obj.relevancy(r, qp)
        qp = obj.query_point(qp)
        i = input('Continue? (1) / Stop (0): ')
