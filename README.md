# Objectif : Réaliser le nettoyage et le prétraitement des données (priorité). A terme créer un système de recommandation de films de base en utilisant un ensemble de données.

**Ensemble de Données** : L'ensemble de données MovieLens (disponible sur GroupLens) est largement utilisé pour les recommandations de films. Il contient des évaluations d'utilisateurs pour des films, des métadonnées sur les films et des informations démographiques sur les utilisateurs.
L'ensemble de données contient des évaluations d'utilisateurs pour des films, des métadonnées sur les films et des informations démographiques sur les utilisateurs.
Quelques suggestions d’ ensemble de données :
Small: 100,000 ratings
https://grouplens.org/datasets/movielens/
https://grouplens.org/datasets/movielens/latest/
Full: approximately 33,000,000 ratings

**Tâches :**
- Charger l'ensemble de données dans un environnement de traitement des données approprié (comme Python ou R).
- Réaliser le nettoyage et le prétraitement des données.
- Mettre en œuvre un algorithme de recommandation simple (comme le filtrage collaboratif).
- Testez l'algorithme en saisissant les préférences cinématographiques d'un utilisateur et en prédisant d'autres films qui pourraient lui plaire



## Tâche 1: Charger l'ensemble de données

Nous allons utiliser Python avec des bibliothèques populaires comme pandas pour charger et manipuler les données. Les fichiers CSV que vous avez fournis (ratings.csv, links.csv, movies.csv, tags.csv) contiennent respectivement les évaluations des utilisateurs, les liens vers des bases de données de films, les informations sur les films, et les tags associés aux films.

Je vais commencer par charger ces fichiers pour examiner leur contenu.

Les données ont été chargées avec succès et voici un aperçu de chaque fichier CSV :

- ratings.csv contient les colonnes userId, movieId, rating, et timestamp, qui représentent respectivement l'identifiant de l'utilisateur, l'identifiant du film, la note attribuée par l'utilisateur au film, et le moment de la notation.
- links.csv contient les colonnes movieId, imdbId, et tmdbId, qui fournissent les identifiants pour chaque film dans les bases de données IMDB et TMDB.
- movies.csv contient les colonnes movieId, title, et genres, qui incluent l'identifiant du film, son titre, et ses genres.
- tags.csv contient les colonnes userId, movieId, tag, et timestamp, qui représentent l'identifiant de l'utilisateur, l'identifiant du film, le tag (étiquette) attribué au film par l'utilisateur, et le moment de l'attribution du tag.

Code : 
```
import pandas as pd

# Charger les fichiers CSV
ratings_df = pd.read_csv('/mnt/data/ratings.csv')
links_df = pd.read_csv('/mnt/data/links.csv')
movies_df = pd.read_csv('/mnt/data/movies.csv')
tags_df = pd.read_csv('/mnt/data/tags.csv')

# Afficher les premières lignes de chaque dataframe pour un aperçu
(ratings_df.head(), links_df.head(), movies_df.head(), tags_df.head())
```


## Tâche 2: Nettoyage et prétraitement des données

Le nettoyage et le prétraitement des données sont cruciaux pour assurer la qualité des recommandations produites par le système. Voici les étapes que nous allons suivre pour cette tâche :

- Vérifier et traiter les valeurs manquantes dans chaque fichier CSV pour s'assurer qu'il n'y a pas de données incomplètes qui pourraient fausser les recommandations.
- Normaliser les genres dans movies.csv pour faciliter leur manipulation lors de la recommandation de films.
- Convertir les timestamps en dates lisibles dans ratings.csv et tags.csv, si nécessaire, pour une meilleure interprétation des données.
- Filtrer les données, si nécessaire, pour supprimer les utilisateurs ou les films avec un faible nombre de notes, car ils peuvent ne pas être suffisamment représentatifs pour le système de recommandation.

Commençons par ces étapes de nettoyage et de prétraitement.

Les résultats du nettoyage initial montrent que :

- Il n'y a pas de valeurs manquantes dans ratings.csv, movies.csv, et tags.csv.
- Dans links.csv, il y a 8 valeurs manquantes dans la colonne tmdbId. Ces valeurs manquantes pourraient ne pas être critiques pour notre système de recommandation de films basé principalement sur les évaluations, les genres, et les tags. Cependant, pour l'intégrité des données, nous pourrions choisir de les ignorer ou d'appliquer une stratégie de traitement spécifique si ces liens étaient nécessaires pour des fonctionnalités spécifiques (comme la récupération d'informations supplémentaires à partir de TMDB).

Normalisation des genres

Pour faciliter le traitement des genres dans movies.csv, nous pouvons transformer la chaîne des genres en une liste de genres distincts. Cela nous permettra de manipuler plus facilement les genres lors de la création des recommandations.
Conversion des timestamps

La conversion des timestamps en dates lisibles n'est pas strictement nécessaire pour la mise en œuvre de l'algorithme de recommandation. Cependant, elle peut être utile pour l'analyse exploratoire ou pour comprendre le comportement des utilisateurs au fil du temps. Nous pouvons décider de reporter cette étape si elle n'est pas pertinente pour notre objectif principal.
Filtrage des données

Cette étape peut nécessiter une analyse supplémentaire pour identifier les seuils appropriés pour filtrer les utilisateurs ou les films avec un faible nombre de notes. Cela dit, procédons d'abord à la normalisation des genres dans movies.csv. 

- 1. Vérifier les valeurs manquantes pour chaque dataframe
missing_values = {
    "ratings_missing": ratings_df.isnull().sum(),
    "links_missing": links_df.isnull().sum(),
    "movies_missing": movies_df.isnull().sum(),
    "tags_missing": tags_df.isnull().sum()
}

- 2. & 3. Pour le moment, nous nous concentrons sur les valeurs manquantes et la normalisation des genres sera traitée ensuite
missing_values

Code :
```
# Normaliser les genres en les transformant en listes
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

# Aperçu des premières lignes pour vérifier la transformation
movies_df.head()
```

## Tache 3 : Réindexation et Création de la Matrice Utilisateur-Film

Nous commençons par créer des mappings pour les IDs de films et d'utilisateurs afin de s'assurer que les indices de notre matrice correspondent de manière séquentielle aux films et aux utilisateurs. Ensuite, nous utiliserons ces mappings pour créer une matrice sparse utilisateur-film.

Code :
```
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Réindexation pour garantir la cohérence
# Création de mappings pour les IDs de films et d'utilisateurs
unique_movies = ratings_df['movieId'].unique()
movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
idx_to_movie = {idx: movie for movie, idx in movie_to_idx.items()}

unique_users = ratings_df['userId'].unique()
user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}

# Création de la matrice sparse correcte
rows = ratings_df['userId'].map(user_to_idx)
cols = ratings_df['movieId'].map(movie_to_idx)
values = ratings_df['rating']

# La forme de la matrice doit correspondre au nombre total d'utilisateurs et de films uniques
movie_user_matrix_sparse = csr_matrix((values, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

movie_user_matrix_sparse.shape
```

La matrice utilisateur-film a été créée avec succès et est de forme 610×9724610×9724, ce qui signifie qu'elle inclut 610 utilisateurs et 9724 films, reflétant la structure de notre ensemble de données.

## Tache 4 : Calcul de la Similarité entre les Films

Maintenant que nous avons notre matrice utilisateur-film correctement formatée, nous allons procéder au calcul de la similarité entre les films en utilisant la similarité cosinus. Pour ce faire, nous utiliserons NearestNeighbors de sklearn, configuré pour utiliser la métrique cosinus. Nous ajusterons ce modèle sur la matrice transposée pour calculer la similarité entre les films, puis nous testerons le système en trouvant les films les plus similaires à "Toy Story".

Code :
```
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Réindexation pour garantir la cohérence
# Création de mappings pour les IDs de films et d'utilisateurs
unique_movies = ratings_df['movieId'].unique()
movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
idx_to_movie = {idx: movie for movie, idx in movie_to_idx.items()}

unique_users = ratings_df['userId'].unique()
user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}

# Création de la matrice sparse correcte
rows = ratings_df['userId'].map(user_to_idx)
cols = ratings_df['movieId'].map(movie_to_idx)
values = ratings_df['rating']

# La forme de la matrice doit correspondre au nombre total d'utilisateurs et de films uniques
movie_user_matrix_sparse = csr_matrix((values, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

movie_user_matrix_sparse.shape
```

Les 10 films les plus similaires à "Toy Story" ont été trouvés avec succès, en utilisant la similarité cosinus pour calculer leur proximité. Voici les films recommandés avec leurs distances respectives par rapport à "Toy Story" (la distance est un indicateur de la similarité, où une distance plus faible signifie une plus grande similarité) :

- Star Wars: Episode IV - A New Hope (1977) - Distance: 0.4274
- Forrest Gump (1994) - Distance: 0.4344
- Lion King, The (1994) - Distance: 0.4357
- Jurassic Park (1993) - Distance: 0.4426
- Mission: Impossible (1996) - Distance: 0.4529
- Independence Day (a.k.a. ID4) (1996) - Distance: 0.4589
- Star Wars: Episode VI - Return of the Jedi (1983) - Distance: 0.4589
- Groundhog Day (1993) - Distance: 0.4611
- Back to the Future (1985) - Distance: 0.4658
- Toy Story 2 (1999) - Distance: 0.4696

Cela conclut notre implémentation de l'algorithme de recommandation simple utilisant le filtrage collaboratif basé sur la similarité cosinus. Les films recommandés semblent être des choix pertinents, étant des films populaires qui partagent des thèmes ou des genres similaires avec "Toy Story".

Code : 
```
# Configuration du modèle KNN avec la similarité cosinus
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

# Ajustement du modèle sur la matrice transposée pour calculer la similarité entre les films
model_knn.fit(movie_user_matrix_sparse.T)  # Transposée pour calculer la similarité entre les films

# Sélection d'un film pour tester - Prenons un film populaire comme exemple, "Toy Story" (movieId 1)
toy_story_idx = movie_to_idx[1]  # Obtenir l'index de "Toy Story" dans notre matrice

# Trouver les 10 films les plus similaires à "Toy Story"
distances, indices = model_knn.kneighbors(movie_user_matrix_sparse.T[toy_story_idx], n_neighbors=11)

# Convertir les indices en titres de films
similar_movies = [idx_to_movie[idx] for idx in indices.flatten()][1:]  # Exclure le premier qui est "Toy Story" lui-même
similar_movies_distances = distances.flatten()[1:]  # Les distances des films similaires

# Récupérer les titres des films similaires à partir de leur movieId
similar_movies_titles = movies_df[movies_df['movieId'].isin(similar_movies)]['title'].values

similar_movies_titles, similar_movies_distances
```
