import pandas as pd

# Chargement des fichiers CSV
ratings_df = pd.read_csv('/chemin/vers/ratings.csv')
links_df = pd.read_csv('/chemin/vers/links.csv')
movies_df = pd.read_csv('/chemin/vers/movies.csv')
tags_df = pd.read_csv('/chemin/vers/tags.csv')

# Affichage des premières lignes pour vérifier le chargement correct des données
print(ratings_df.head())
print(links_df.head())
print(movies_df.head())
print(tags_df.head())
