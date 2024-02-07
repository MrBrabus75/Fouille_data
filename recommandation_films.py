from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Configuration et ajustement du modèle KNN avec similarité cosinus
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(movie_user_matrix_sparse.T)  # Transposée pour calculer la similarité entre les films

# Sélection d'un film et recherche des films les plus similaires
toy_story_idx = movie_to_idx[1]  # Index de "Toy Story"
distances, indices = model_knn.kneighbors(movie_user_matrix_sparse.T[toy_story_idx], n_neighbors=11)

# Affichage des films recommandés en excluant le premier résultat (Toy Story lui-même)
films_recommandes = [idx_to_movie[idx] for idx in indices.flatten()][1:]
print("Films recommandés:", films_recommandes)