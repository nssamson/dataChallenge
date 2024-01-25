#%%

#############LASSSSSO


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
from sklearn.metrics import mean_absolute_error
from math import sqrt

# on prend l'erreur absolue moyenne


import importlib

import featuring as ft
importlib.reload(ft)

#%%

path_to_data = "data"
# Load files
x_trainT = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)
y_trainT = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)
x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)

le = LabelEncoder()

x_test["type_territoire"] = le.fit_transform(x_test["type_territoire"])
x_trainT["type_territoire"] = le.fit_transform(x_trainT["type_territoire"])


# on va extraire un sous ensemble de données pour tester le modèle
# on va prendre 10% des données pour tester le modèle

x_train, x_eval, y_train, y_eval = train_test_split(x_trainT, y_trainT, test_size=0.2, random_state=0)




#%%

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators =10, random_state=0, max_depth=5, criterion="absolute_error"
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        learning_rate = 0.2, max_depth= 19, min_samples_leaf= 9, min_samples_split= 10, n_estimators= 196, subsample= 1, random_state=0, loss='absolute_error'),
    "NeutralGradientBoostingRegressor": GradientBoostingRegressor(random_state=0, loss='absolute_error'),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        max_iter = 80, max_depth= 5, min_samples_leaf= 3, learning_rate=0.1 , random_state=0, loss='absolute_error'),
    "NeutralHistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=0, loss='absolute_error'),
    "linearregression": LinearRegression(),
    "lasso": Lasso(alpha=10000, max_iter=500, tol=0.001),
    "lassoCv": LassoCV(alphas=[ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 20000, 30000, 100000, 10000000 ], max_iter=500, tol=0.001, cv = 5, verbose=2),
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
x_trainT = data_prep_pipeline.fit_transform(x_trainT)
x_train = data_prep_pipeline.fit_transform(x_train)
# on applique le pipeline sur x_test
x_test = data_prep_pipeline.transform(x_test)
x_eval = data_prep_pipeline.transform(x_eval)

#%%

def get_estimator(model_name="lasso", is_lin=True, is_scale=True):

    lab_processor = LabelEncoder()
    if (is_lin):
        cat_processor = OneHotEncoder()
    else:
        cat_processor = OrdinalEncoder()
    if (is_scale):
        num_processor = StandardScaler()
    else:
        num_processor = "passthrough"
    
    # on veut toutes les colonnes numériques
    num_columns = x_train.select_dtypes(include=np.float64).columns.tolist()
    categ_features = []
    suppr_features = ["id_poste_source", "id_dr", 
                    # 'saison', 'year',
                    'type_territoire'
                    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num',num_processor, num_columns),
            ('encoding', cat_processor, categ_features),
            ('drop_columns', 'drop', suppr_features)
        ]
    )
    
    pipeline = Pipeline([
        ('proc', preprocessor),
        ('mod', models[model_name])
    ])
    
    return pipeline





#%%

def evalModel (vmodel, v_splits = 5) : 
    
    ss = ShuffleSplit(n_splits=v_splits, test_size=0.2, random_state=0)

    # Appliquez votre modèle pré-entraîné sur chaque sous-ensemble
    for i, (train_index, test_index) in enumerate(ss.split(x_train)):
        print(f'Sous-ensemble {i+1}')
        X1, y1 = x_train.iloc[test_index], y_train.iloc[test_index]
        # Faites des prédictions sur le sous-ensemble de test
        predictions = vmodel.predict(X1)
        # Évaluez la performance du modèle sur le sous-ensemble de test (par exemple, avec la MAE)
        mae = mean_absolute_error(y1, predictions)
        print(f'Sous-ensemble {i+1} MAE: {mae:.4f}')
        # affiche la taille des sous ensembles
        print(f'Sous-ensemble {i+1} taille test: {len(test_index)}')
        print(f'Sous-ensemble {i+1} taille train: {len(train_index)}\n')

        
# Récupération de l'importance des features
def get_feature_importance(vmodel):

    # on test si le modèle a une fonction feature_importances_
    if not hasattr(vmodel.named_steps['mod'], 'feature_importances_'):

        #on teste si le modele a des coefficients
        if not hasattr(vmodel.named_steps['mod'], 'coef_'):
            
            print("Le modèle n'a pas de fonction feature_importances_ ni de coefficients")
            return None
        
        else:
                    
            # on étudie les coefficients de la regression linéaire

            lasso_coefficients = vmodel.named_steps['mod'].coef_
            # Récupérer les noms des caractéristiques après le prétraitement
            columns_after_transform = vmodel.named_steps['proc'].get_feature_names_out()
            # Créer un DataFrame avec les noms des caractéristiques et les coefficients associés
            coefficients_df = pd.DataFrame({'Feature': columns_after_transform, 'Coefficient': lasso_coefficients})
            # trier selon les valeurs absolues des coefficients
            coefficients_df['AbsCoefficient'] = coefficients_df['Coefficient'].abs()
            coefficients_df = coefficients_df.sort_values(by='AbsCoefficient', ascending=False)
            # Afficher les coefficients
            print('Coefficients du modèle LASSO avec noms de caractéristiques :\n', coefficients_df)

            return coefficients_df
    
    else :
        # on doit récupérer le modèle/ feature importance
        feature_importances = vmodel.named_steps['mod'].feature_importances_
        column_transformer = vmodel.named_steps['proc']
        col = column_transformer.get_feature_names_out()
        # Création d'un DataFrame pour afficher les importances
        feature_importance_df = pd.DataFrame({'Feature': col, 'Importance': feature_importances})
        # Trier le DataFrame par importance descendante
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        # Affichage des 10 premières features les plus importantes
        print(feature_importance_df.head(100))

        # Tracé d'un graphique des importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
        plt.xlabel('Importance')
        plt.title('Top 20 des importances des features dans le modèle RandomForest')
        plt.show()
        
        return feature_importance_df

def analyseModel(vmodel, v_splits = 3):
    
    print ("Analyse du modèle :")
    # on veut rappeler les caracétistiques du modèle
    print(vmodel.named_steps['mod'])
    print("\n")

    # Calculate the mean absolute error on the training data
    y_pred_train = vmodel.predict(x_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    print("MAE total x_train:", mae)
    y_pred_eval = vmodel.predict(x_eval)
    mae = mean_absolute_error(y_eval, y_pred_eval)
    print("MAE total x_eval:", mae)

    evalModel(vmodel,v_splits)
    
    return get_feature_importance(vmodel)




#%%

#%%

############################################"
# STOP"


modelL = get_estimator('lasso',is_scale=True, is_lin=True)

modelL.fit(x_train, y_train)

analyseModel(modelL, v_splits=3)

#%%
#%%


## on prépare une cross validation
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=0)
scores = cross_val_score(modelL, x_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
print(scores)
print(np.mean(scores))



#%%

coefficients_dfL = analyseModel(modelL, v_splits=3)
# quelles sont les colonnes qui ont été supprimées par le lasso

# on récupère les colonnes qui ont un coefficient nul
columns_to_drop = coefficients_dfL[coefficients_dfL['Coefficient'] == 0]['Feature'].tolist()
columns_to_drop

# on récupère les 12 premières colonnes
columns_to_drop2 = coefficients_dfL['Feature'].tolist()[11:]
columns_to_drop2
##%
##################################

# RANDOM FOREST

##################################

#%%

modelRF = get_estimator('RandomForest',is_scale=False, is_lin=False)

modelRF.fit(x_train, y_train)

analyseModel(modelRF, v_splits=3)



#%%

# Allez, on va essayer de faire un modèle avec le gradient boosting

modelGB = get_estimator('GradientBoostingRegressor',is_scale=False, is_lin=False)

modelGB.fit(x_train, y_train)

analyseModel(modelGB, v_splits=3)

#%%

# cross validation de la GB

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
scores = cross_val_score(modelGB, x_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
print(scores)


#%%

# On cherche les meilleurs coefficients pour le gradient boosting
###############################################################

from sklearn.model_selection import GridSearchCV

# pour les données de cross validation
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    return cv.split(X)

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

param_distRF = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': [2, 3, 4, 5, 10, 30],
    'min_samples_leaf': [1, 2, 3, 4, 5, 10],
    'max_features': [0.5, 0.7, 1.0],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'random_state': [None, 42],
    'ccp_alpha': [0, 0.1, 0.2],
    'max_samples': [None, 0.5, 0.8, 1.0]
}
    
# Créer une instance de GridSearchCV avec votre pipeline et la grille de paramètres
# grid_search = GridSearchCV(get_estimator2(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

rand_search = RandomizedSearchCV(
    get_estimator('RandomForest',is_lin=False, is_scale=False), param_distributions=param_distRF,
    n_iter=20, n_jobs=-1, cv=lecv, random_state=0, verbose = 2, scoring='neg_mean_absolute_error'
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
results.to_csv("coef GradientB results.csv")
#%%
rand_search.best_params_

#%%
# analyser les coefficients de la GB






#%%

## du coup on calcule sur tout le jeu d'entrainement

modelGBT = get_estimator('GradientBoostingRegressor',is_scale=False, is_lin=False)

modelGBT.fit(x_trainT, y_trainT)

analyseModel(modelGBT, v_splits=3)

#%%

#%%
predicted_values = modelGBT.predict(x_test)

y_pred = pd.DataFrame(data=predicted_values, index=x_test.index.values, columns=["pertes_totales"])
y_pred.index.name = 'ID'

y_pred.to_csv(path_to_data + "/testing_modelGB.csv")