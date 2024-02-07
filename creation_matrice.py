from scipy.sparse import csr_matrix
import pandas as pd

# Mappings pour les IDs de films et d'utilisateurs
unique_movies = ratings_df['movieId'].unique()
movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
unique_users = ratings_df['userId'].unique()
user_to_idx = {user: idx for idx, user in enumerate(unique_users)}

# Création de la matrice sparse utilisateur-film
rows = ratings_df['userId'].map(user_to_idx)
cols = ratings_df['movieId'].map(movie_to_idx)
values = ratings_df['rating']
movie_user_matrix_sparse = csr_matrix((values, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

# Affichage de la forme de la matrice
print(movie_user_matrix_sparse.shape)