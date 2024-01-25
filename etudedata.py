#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from datetime import timedelta
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

#%%

# c'est parti pour le preprocessing

#%%
# on importe les données

# pour les données de cross validation
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X)

path_to_data = "data"
# Load files
x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)
y_train = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)
x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)


lecv = get_cv(x_train, y_train)


le = LabelEncoder()

x_test["type_territoire"] = le.fit_transform(x_test["type_territoire"])
x_train["type_territoire"] = le.fit_transform(x_train["type_territoire"])

#%%

x_train.info()  

x_train
#%%
y_train


# on joint x_train et y_train pour pouvoir faire des traitements sur les données
# on joint sur l'index 
# x_train.join(y_train)

tX = x_train.join(y_train) 

#  x_train = x_train.drop(columns=["id_dr", "id_poste_source"])

#%%

# dans tX on trace les courbes de conso_totale et prod_totale
tX.plot(y=["conso_totale", "prod_totale"])

# on trace la somme production + consommation
# on crée une colonne somme_prod_conso
tX['diff_prod_conso'] = tX['conso_totale'] + tX['prod_totale']
tX.plot(y=["diff_prod_conso"])

# on trace un graph avec diff_prod_conso en abscisse et pertes_totales en ordonnée
tX.plot.scatter(x='diff_prod_conso', y='pertes_totales')

#%%

# on trace sur le meme graphe les courbes de conso_totale en rouge et prod_totale en bleu
# sur un meme graphique
# on ordonne tX par val de conso_totale decroissante
tXs = tX.sort_values(by=['conso_totale'], ascending=False)
#on réindex tXs
tXs = tXs.reset_index(drop=True)
# on trace les courbes
tXs.plot(y=["conso_totale", "prod_totale", "pertes_totales"])

#%%

#On veut connaitre la répartition des valeurs de conso_totale
# on va faire un histogramme
tX['conso_totale'].hist(bins=100)

# on veut connaitre la répartition des valeurs de prod_totale
# on va faire un histogramme
tX['prod_totale'].hist(bins=100)

# on veut connaitre la répartition des valeurs de pertes_totales

# on va faire un histogramme
tX['pertes_totales'].hist(bins=100)

#%%

# filtrons sur les pertes totales < 0
tXp = tX[tX['pertes_totales'] < 0]

# regardons les taux linky dans tXp 
tXp['taux_linky'].hist(bins=100)

# on trace tauxlinky en fonction de pertes_totales
tX.plot.scatter(x='taux_linky', y='pertes_totales')

# on crée une colonne taux_linky * conso_totale
tX['conso_linky'] = tX['conso_totale'] * tX['taux_linky'] /100
#
tX.plot.scatter(x='conso_linky', y='pertes_totales')



#%%
tX.plot(y="conso_totale", color="red")
tX.plot(y="prod_totale", color="blue")
tX.plot(y="pertes_totales", color="green")


#%%

# 

# 
# quelle est la moyenne des pertes totales de y_train?

y_train.mean()

# on fait un vecteur de la taile de x_test avec la moyenne des pertes totales
y_test = np.full((len(x_test), 1), y_train.mean())

# on crée un dataframe avec y_test
y_test = pd.DataFrame(y_test, columns=['pertes_totales'])

# on fait un vecteur de la taille de y_train avec la moyenne des pertes totales
y_pred = np.full((len(y_train), 1), y_train.mean())

# on crée un dataframe avec y_train
y_pred = pd.DataFrame(y_train, columns=['pertes_totales'])

# On calcule la MEA
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_train, y_pred)

from sklearn.dummy import DummyRegressor

# Créer un objet DummyRegressor
dummy_regressor = DummyRegressor(strategy='mean')

# Entraîner le modèle sur les données d'entraînement
dummy_regressor.fit(x_train, y_train)

# Faire des prédictions sur les données de test
y_pred_dummy = dummy_regressor.predict(x_test)

# Calculer la MAE sur les données de test
mean_absolute_error(y_test, y_pred_dummy)

#%%
# on fait une cross validation 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dummy_regressor, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(scores)





#%%

# calculer la moyenne pour chaque colonne numérique de x_train
x_train.describe()
# on prend la ligne des moyennes de x_train.describe()
x_train.describe().iloc[1]

# on trouve les colonnes de x_train qui ne se terminent pas par un '_2'
col_moy = []
for i in x_train.columns:
    if (i[-2:] != "_2") and (x_train[i].dtype == "float64"):
        col_moy.append(i)
col_moy

# on trouve les moyennes par colonnes de col_moy dans x_train 
x_train[col_moy].mean()
# sur un graphique
x_train[col_moy].mean().plot.bar()





# x_train.mean()


#%%

# Il faudrait trouver les valeurs abhérentes dans x_train parrapport à x_test
# on va utiliser la fonction isin()
# on va faire une boucle sur les colonnes de x_train
# on va comparer les valeurs de x_train et x_test
# on va faire une liste des valeurs abhérentes
# on va faire une boucle sur les colonnes de x_train
# on va comparer les valeurs de x_train et x_test
# on va faire une liste des valeurs abhérentes

# on crée une liste vide
liste_valeurs_abh = []
# on crée une liste vide
liste_colonnes = []

#%%
#comp

# donne les statistiques sur x_test
print(x_test.describe())
# sur les seules colonnes conso_totale et prod_totale
print(x_test[["conso_totale", "prod_totale"]].describe())
# pareil sur x_train
print(x_train[["conso_totale", "prod_totale"]].describe())

# on trace sur une courbe les valeurs de conso_totale et prod_totale de x_test en bleu histograme

x_test[["conso_totale", "prod_totale"]].hist(bins=100)
x_train[["conso_totale", "prod_totale"]].hist(bins=100)

# donne les statistiques sur y_train
print(y_train.describe())
# sur la seule colonne pertes_totales


#%%
# Create empty lists
outlier_values = []
outlier_columns = []

# Loop through the columns of x_train
for column in x_train.columns:

    # testons si la colonne est numérique
    if x_train[column].dtype == "float64":
    # donne moi les valeurs max et min de x_test pour cette colonne
        max_test = x_test[column].max()
        min_test = x_test[column].min()
        # donne moi en outliers les valeurs de x_train qui ne sont pas dans l'intervalle [min_test, max_test]    
        outliers = x_train[~x_train[column].between(min_test, max_test)]    
        # Check if there are any outliers
        if not outliers.empty:
            # Append the outlier values and column name to the lists
            outlier_values.append(outliers[column].values)
            outlier_columns.append(column)

# Print the outlier values and columns
for values, column in zip(outlier_values, outlier_columns):
    print(f"Outliers in column '{column}': {values}")


#%%
    
# donne moi la distribution des pertes de y_train
y_train.hist(bins=100)
#%%
# compte les pertes négatives et positives
print(y_train[y_train["pertes_totales"] < 0].count())
print(y_train[y_train["pertes_totales"] > 0].count())
