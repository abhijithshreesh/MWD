import pandas as pd

from phase_3_task_1 import UserMovieRecommendation


class MovieLSH(UserMovieRecommendation):

    def __init__(self):
        super().__init__()
        self.movie_tag_df = self.get_movie_tag_matrix()
        (self.U, self.s, self.Vt) = self.util.SVD(self.movie_tag_df.values)

    def assign_group(self, value, group_range):
        for i in range(0,len(group_range)):
            if value>=group_range[i] and value<group_range[i+1]:
                return i

    def group_data(self):
        U_dataframe = pd.DataFrame(self.U)
        group_length = {}
        column_groups = {}
        for column in U_dataframe:
            group_length[column] = (U_dataframe[column].max()- U_dataframe[column].min())/10
            column_groups[column] = []
        for column in U_dataframe:
            sum = U_dataframe[column].min()
            for i in range(0, 11):
                column_groups[column].append(sum)
                sum = sum+group_length[column]
            column_groups[column][-1] += 1
        interval_df = pd.DataFrame()
        for column in U_dataframe:
            interval_df[column] = pd.Series([self.assign_group(each, column_groups[column]) for each in U_dataframe[column]],
                                  index=U_dataframe.index)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='phase_2_task_1a.py 146',
    # )
    # parser.add_argument('user_id', action="store", type=int)
    # input = vars(parser.parse_args())
    # user_id = input['user_id']
    obj = MovieLSH()
    obj.group_data()