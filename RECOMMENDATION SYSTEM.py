pip install pandas numpy scikit-learn

import pandas as pd
import numpy as np

movies = pd.read_csv('movies.csv')  # Movie titles
ratings = pd.read_csv('ratings.csv')  # Ratings given by users


print(movies.head())
print(ratings.head())


movie_ratings = pd.merge(ratings, movies, on='movieId')


movie_ratings = movie_ratings[['userId', 'title', 'rating']]


print(movie_ratings.head())


user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')


user_movie_matrix.fillna(0, inplace=True)

print(user_movie_matrix.head())

from sklearn.metrics.pairwise import cosine_similarity


movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)


print(movie_similarity_df.head())

def get_movie_recommendations(movie_title, similarity_matrix, num_recommendations=5):
    
    similar_scores = similarity_matrix[movie_title].sort_values(ascending=False)

    
    recommendations = similar_scores.iloc[1:num_recommendations+1]

    return recommendations


recommended_movies = get_movie_recommendations('thriller Story (2000)', movie_similarity_df, num_recommendations=5)
print("Movies recommended based on 'thriller Story (2000)':")
print(recommended_movies)

def recommend_movies_for_user(user_id, user_movie_matrix, similarity_matrix, num_recommendations=5):
    user_ratings = user_movie_matrix.loc[user_id]

    
    watched_movies = user_ratings[user_ratings > 0].index.tolist()

    
    recommendations = pd.Series(dtype='float64')

    for movie in watched_movies:
        similar_movies = get_movie_recommendations(movie, similarity_matrix, num_recommendations)
        recommendations = recommendations.append(similar_movies)

   
    recommendations = recommendations.groupby(recommendations.index).mean().sort_values(ascending=False)

    
    recommendations = recommendations.drop(watched_movies, errors='ignore')

    return recommendations.head(num_recommendations)

user_recommendations = recommend_movies_for_user(1, user_movie_matrix, movie_similarity_df, num_recommendations=5)
print(f"Movies recommended for User 1:")
print(user_recommendations)

