from collections import Counter

from scripts.phase2.common.config_parser import ParseConfig
from scripts.phase2.common.data_extractor import DataExtractor


class UserMovieRecommendation(object):
    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.reshuffle = False

    def get_highest_percentage_genre(self, genre_counter):
        result_genre = ""
        max_count = -1
        for genre in genre_counter:
            if max_count == -1 or genre_counter[genre] >= max_count:
                result_genre = genre
                max_count = genre_counter[genre]

        if max_count == -1:
            return -1

        return result_genre

    def get_least_percentage_genre(self, genre_counter):
        result_genre = ""
        least_count = 0
        for genre in genre_counter:
            if least_count == 0 or genre_counter[genre] <= least_count:
                result_genre = genre
                least_count = genre_counter[genre]

        return result_genre

    def get_combined_data(self):
        mltags = self.data_extractor.get_mltags_data()
        mlmovies = self.data_extractor.get_mlmovies_data()
        result = mltags.merge(mlmovies, left_on="movieid", right_on="movieid", how="left")
        del result['timestamp']
        del result['tagid']
        del result['year']

        return result

    def get_all_movies_for_genre(self, genre):
        mlmovies = self.data_extractor.get_mlmovies_data()
        genre_data = mlmovies[mlmovies['genres'].str.contains(genre)]
        movies = genre_data['moviename'].unique()

        return movies

    def get_all_movies_for_user(self, user_id):
        combined_data = self.get_combined_data()
        user_data = combined_data[combined_data['userid'] == user_id]
        movies = user_data['moviename'].unique()

        return movies

    def get_movie_recommendation(self, genre, user_id, recommended_movies):
        genre_movies = self.get_all_movies_for_genre(genre)
        user_watched_movies = self.get_all_movies_for_user(user_id)
        result = (set(genre_movies) - set(user_watched_movies) - set(recommended_movies))

        if len(result) != 0:
            return result.pop()
        else:
            return "~~NOT-FOUND~~"

    def reshuffle_movie_recommendation(self, genre_counter, movie_recommendation_counter, genre):
        count = movie_recommendation_counter[genre]
        del movie_recommendation_counter[genre]
        next_best_genre = self.get_highest_percentage_genre(movie_recommendation_counter)

        if next_best_genre == -1:
            if self.reshuffle:
                return None
            self.reshuffle = True
            movie_recommendation_counter = {}
            new_genre = ""
            for genre in genre_counter:
                new_genre = genre
                if genre_counter[genre] == 0:
                    movie_recommendation_counter[genre] = 0
            movie_recommendation_counter[new_genre] = count
        else:
            movie_recommendation_counter[next_best_genre] += count

        return movie_recommendation_counter

    def recommendedMovies(self, user_id):
        recommended_movies = []
        combined_data = self.get_combined_data()
        user_data = combined_data[combined_data['userid'] == user_id]

        genre_counter = Counter()
        total_genres_count = 0
        for index, row in user_data.iterrows():
            genres = row['genres'].split("|")
            for genre in genres:
                genre_counter[genre] += 1
                total_genres_count += 1

        if total_genres_count == 0:
            print("THIS USER HAS NOT WATCHED ANY MOVIE")
            exit(1)

        genre_counter_copy = genre_counter.copy()
        total_movies_count = 0
        for genre in genre_counter:
            genre_counter[genre] /= float(total_genres_count)
            if genre_counter[genre] <= 0.15:
                genre_counter[genre] = 0
            genre_counter[genre] *= 5
            genre_counter[genre] = round(genre_counter[genre])
            total_movies_count += genre_counter[genre]

        if total_movies_count == 0:
            # pick the highest genre and recommend movies from that genre
            highest_percentage_genre = self.get_highest_percentage_genre(genre_counter_copy)
            genre_counter[highest_percentage_genre] = 5
        elif total_movies_count > 5:
            # remove movie from the genre that has the least percentage
            least_percentage_genre = self.get_least_percentage_genre(genre_counter_copy)
            genre_counter[least_percentage_genre] -= (total_movies_count - 5)
        elif total_movies_count < 5:
            # add movie to the genre that has the best percentage
            highest_percentage_genre = self.get_highest_percentage_genre(genre_counter_copy)
            genre_counter[highest_percentage_genre] += (5 - total_movies_count)

        movie_recommendation_counter = {}
        for genre in genre_counter:
            if genre_counter[genre] != 0:
                movie_recommendation_counter[genre] = genre_counter[genre]

        loop_counter = 0
        while len(recommended_movies) != 5:
            loop_counter += 1
            if loop_counter == 100:
                print("UNABLE TO FIND MORE MOVIES FOR RECOMMENDATION")
                break

            genre = self.get_highest_percentage_genre(movie_recommendation_counter)
            movie = self.get_movie_recommendation(genre, user_id, recommended_movies)
            if movie != "~~NOT-FOUND~~":
                movie_recommendation_counter[genre] -= 1
                recommended_movies.append(movie)
            else:
                print("Reshuffling needed")
                movie_recommendation_counter = self.reshuffle_movie_recommendation(genre_counter,
                                                                                   movie_recommendation_counter, genre)
                if movie_recommendation_counter is None:
                    print("UNABLE TO FIND MORE MOVIES FOR RECOMMENDATION")
                    break

        return recommended_movies


if __name__ == "__main__":
    obj = UserMovieRecommendation()
    user_id = 146
    print("Movie recommendation for user id " + str(user_id))
    movies = obj.recommendedMovies(user_id)
    for movie in movies:
        print(movie)
