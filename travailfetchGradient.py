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
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import importlib

import featuring as ft
#%%
# rafraichir l'import de featuring

importlib.reload(ft)

#%%


#%%

path_to_data = "data"
# Load files
x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)
y_train = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)
x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)

le = LabelEncoder()

x_test["type_territoire"] = le.fit_transform(x_test["type_territoire"])
x_train["type_territoire"] = le.fit_transform(x_train["type_territoire"])


#%%


models = {
    "RandomForest": RandomForestRegressor(
        n_estimators = 10, random_state=0, max_depth=5, criterion="absolute_error"
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        learning_rate = 0.2, max_depth= 14, min_samples_leaf= 6, min_samples_split= 5, n_estimators= 90, subsample= 1, random_state=0, loss='absolute_error'),
    "NeutralGradientBoostingRegressor": GradientBoostingRegressor(random_state=0, loss='absolute_error'),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        max_iter = 80, max_depth= 5, min_samples_leaf= 3, learning_rate=0.1 , random_state=0, loss='absolute_error'),
    "NeutralHistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=0, loss='absolute_error'),
    "linearregression": LinearRegression(),
    "lasso": Lasso(alpha=0.5),
    "lassoCv": LassoCV(alphas=[10000, 20000, 30000, 100000, 10000000 ], max_iter=20, tol=0.01, cv = 5),
}

#%%

# on fait un pipeline avec toutes les transformations de préparation

data_prep_pipeline = Pipeline([
    ('null', ft.valnulles_encoder), 
    ('date', ft.date_encoder),
    ('prop', ft.prop_encoder), 
    ('poly', ft.puissance_encoder),
    ('purge', ft.purge_data)
    
    # Ajoutez d'autres étapes de préparation des données au besoin
])

# on applique le pipeline sur x_train
x_train = data_prep_pipeline.fit_transform(x_train)
# on applique le pipeline sur x_test
x_test = data_prep_pipeline.transform(x_test)


#%%

#%%

def get_estimator3():

    lab_processor = LabelEncoder()
    cat_processor = OrdinalEncoder()

    num_processor = "passthrough"
    # on veut toutes les colonnes numériques
    num_columns = x_train.select_dtypes(include=np.float64).columns.tolist()
    categ_features = ["month"]
    suppr_features = ["id_poste_source", "id_dr"]

    preprocessor = ColumnTransformer(
        transformers=[
            # ('label_encoding', lab_processor, obj_features),
            ('num','passthrough', num_columns),
            ('ordinal_encoding', cat_processor, categ_features),
            ('drop_columns', 'drop', suppr_features)
        ]
    )
    
    pipeline = Pipeline([
        # ('prep', data_prep_pipeline),
        ('proc', preprocessor),
        ('mod', models["GradientBoostingRegressor"])
    ])
    
    return pipeline

#%%


#%%


model = get_estimator3()

model.fit(x_train, y_train)

#%%

y_pred = model.predict(x_test)

#%%

# on calcule l'erreur sur le jeu d'entrainement
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the mean absolute error on the training data
y_pred_train = model.predict(x_train)
mae = mean_absolute_error(y_train, y_pred_train)
print("MAE:", mae)

#%%
# Missing values
# on remplace les valeurs manquantes par la moyenne pour les seules colonnes numériques de
# x_train

x_train.info()
#%%

#%%


#%%

from sklearn.model_selection import GridSearchCV


lecv = get_cv(x_train, y_train)

# Définir la grille de paramètres pour le GradientBoostingRegressor

param_dist = {
    'mod__n_estimators': randint(10, 200),
    'mod__learning_rate': [0.2, 0.1, 0.01, 0.001],
    'mod__max_depth': randint(5, 20),
    'mod__min_samples_leaf': randint(2, 10),
    'mod__min_samples_split': [2, 3, 4, 5, 10, 30],
    'mod__subsample': [0.8, 1.0]
}

param_distHGBR = {
    'mod__learning_rate': [0.2, 0.1, 0.01, 0.001],
    'mod__max_depth': randint(5, 20),
    'mod__min_samples_leaf': randint(2, 10),
    'mod__max_iter': randint(10, 200)
}

# Créer une instance de GridSearchCV avec votre pipeline et la grille de paramètres
# grid_search = GridSearchCV(get_estimator2(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

rand_search = RandomizedSearchCV(
    get_estimator3(), param_distributions=param_dist,
    n_iter=10, n_jobs=-1, cv=lecv, random_state=0, verbose = 2, scoring='neg_mean_absolute_error'
)

# Appliquer la recherche sur les hyperparamètres sur vos données
rand_search.fit(x_train, y_train)

# Afficher les meilleurs paramètres et le meilleur score
print("Meilleurs paramètres :", rand_search.best_params_)
print("Meilleur score :", rand_search.best_score_)

# on veut voir tous les résultats pour chaque combinaison de paramètres
# classés du meilleur au moins bon

results = pd.DataFrame(rand_search.cv_results_)
results = results.sort_values(by='rank_test_score')
results

#%%
rand_search.best_params_
#%%
#extraire en csv ces résultats
results.to_csv("coef GradientB results.csv")


#%%
#%%
# on calcule l'erreur sur le jeu d'entrainement
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# Make predictions on the test data
y_pred = rand_search.predict(x_test)

# Calculate the mean absolute error on the training data
y_pred_train = rand_search.predict(x_train)
mae = mean_absolute_error(y_train, y_pred_train)
print("MAE:", mae)


#%%

## on prépare une cross validation
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
scores = cross_val_score(get_estimator3(), x_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
print(scores)

#%%

# on analyse les résultats de la cross validation
# on calcule la moyenne des scores
print("Moyenne des scores :", scores.mean())

# on calcule l'écart type des scores
print("Ecart type des scores :", scores.std())

# on calcule l'intervalle de confiance à 95%
print("Intervalle de confiance :", scores.mean() - 2 * scores.std(), scores.mean() + 2 * scores.std())


#%%


#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Récupération de l'importance des features

# on doit récupérer le modèle/ feature importance
feature_importances = model.named_steps['mod'].feature_importances_

# on récupère le nom des colonnes dans le pipeline


column_transformer = model.named_steps['proc']
col = column_transformer.get_feature_names_out()

# col = model.named_steps['proc'].transformers_[0][2] + model.named_steps['proc'].transformers_[1][2]

col

# Création d'un DataFrame pour afficher les importances
feature_importance_df = pd.DataFrame({'Feature': col, 'Importance': feature_importances})

# Trier le DataFrame par importance descendante
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Affichage des 10 premières features les plus importantes
print(feature_importance_df.head(100))

#%%
# Tracé d'un graphique des importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
plt.xlabel('Importance')
plt.title('Top 20 des importances des features dans le modèle RandomForest')
plt.show()
