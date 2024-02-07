import pandas as pd
#from charger_donnees import movies_df
#from charger_donnees import ratings_df
#from charger_donnees import links_df
#from charger_donnees import tags_df

# Supposons que movies_df est déjà chargé
# Normalisation des genres dans movies.csv
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

# Affichage pour vérification
print(movies_df.head())

# Vérification des valeurs manquantes pour chaque DataFrame
print(ratings_df.isnull().sum())
print(links_df.isnull().sum())
print(movies_df.isnull().sum())
print(tags_df.isnull().sum())