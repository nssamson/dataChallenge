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
import holidays
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

#%%

path_to_data = "data"
# Load files
x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)
y_train = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)
x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)


#%%
lecv = get_cv(x_train, y_train)

#%%


#%%


x_train.isnull().sum(axis = 0)

#%%
x_train.info()

#%% traitement des données nulles


def _encode_valnulles(X):
    X_encoded = X.copy()
    # Make sure that DateOfDeparture is of datetime format
    # il nous faut le ratio de conso_reseau_BT sur conso_totale
    X_encoded['ratio_conso_BT'] = X_encoded['conso_reseau_BT'] / X_encoded['conso_totale']  
    # il nous faut le ratio de conso_reseau_HTA sur conso_totale
    X_encoded['ratio_conso_HTA'] = X_encoded['conso_reseau_HTA'] / X_encoded['conso_totale']
    # il nous faut le ratio de prod_reseau_BT sur prod_totale
    X_encoded['ratio_prod_BT'] = X_encoded['prod_reseau_BT'] / X_encoded['prod_totale']
    # il nous faut le ratio de prod_reseau_HTA sur prod_totale
    X_encoded['ratio_prod_HTA'] = X_encoded['prod_reseau_HTA'] / X_encoded['prod_totale']
    # il nous faut le ratio de conso_clients_RES sur conso_totale
    X_encoded['ratio_conso_RES'] = X_encoded['conso_clients_RES'] / X_encoded['conso_totale']
    # il nous faut le ratio de conso_clients_PRO sur conso_totale
    X_encoded['ratio_conso_PRO'] = X_encoded['conso_clients_PRO'] / X_encoded['conso_totale']
    # il nous faut le ratio de conso_clients_ENT sur conso_totale
    X_encoded['ratio_conso_ENT'] = X_encoded['conso_clients_ENT'] / X_encoded['conso_totale']
    # il nous faut le ration de prod_filiere_eolien sur prod_totale
    X_encoded['ratio_prod_eolien'] = X_encoded['prod_filiere_eolien'] / X_encoded['prod_totale']
    # il nous faut le ration de prod_filiere_PV sur prod_totale
    X_encoded['ratio_prod_PV'] = X_encoded['prod_filiere_PV'] / X_encoded['prod_totale']
    # il nous faut le ration de prod_filiere_autre sur prod_totale
    X_encoded['ratio_prod_autre'] = X_encoded['prod_filiere_autre'] / X_encoded['prod_totale']
    

    colonnes_a_remplir = ['prop_clts_logement_indiv', 'prop_clts_logement_collectif',
                        'temperature',
                        'long_reseau_aerien_bt', 'long_reseau_souterrain_bt', 
                        'nb_postes_htabt',
                        'puissance_transfos',
                        'prop_hta_type_1', 'prop_hta_type_2', 'prop_hta_type_3', 'prop_hta_type_4',
                        'ratio_conso_BT', 'ratio_conso_HTA', 'ratio_prod_BT', 'ratio_prod_HTA', 'ratio_conso_RES', 'ratio_conso_PRO', 'ratio_conso_ENT', 'ratio_prod_eolien', 'ratio_prod_PV', 'ratio_prod_autre'
    ]

    # Calcul de la moyenne pour chaque colonne spécifiée
    moyennes_par_colonne = X_encoded[colonnes_a_remplir].mean()
    # Remplissage des valeurs manquantes uniquement dans les colonnes spécifiées avec la moyenne respective
    X_encoded[colonnes_a_remplir] = X_encoded[colonnes_a_remplir].fillna(value=moyennes_par_colonne)

    # on remplit les colonnes suivantes avec une formule
    #conso_reseau_BT ,conso_reseau_HTA,conso_clients_RES ,conso_clients_PRO ,conso_clients_ENT,prod_reseau_BT,
    #prod_reseau_HTA,prod_filiere_eolien,prod_filiere_PV ,
    #prod_filiere_autre

    colonnes_a_remplir2 = ['conso_reseau_BT', 'conso_reseau_HTA', 'conso_clients_RES', 'conso_clients_PRO', 'conso_clients_ENT', 'prod_reseau_BT', 'prod_reseau_HTA', 'prod_filiere_eolien', 'prod_filiere_PV', 'prod_filiere_autre']
    # mais seulement pour les valeurs nulles
    X_encoded['conso_reseau_BT'] = X_encoded['conso_reseau_BT'].fillna(value=X_encoded['conso_totale'] * X_encoded['ratio_conso_BT'])
    X_encoded['conso_reseau_HTA'] = X_encoded['conso_reseau_HTA'].fillna(value=X_encoded['conso_totale'] * X_encoded['ratio_conso_HTA'])
    X_encoded['conso_clients_RES'] = X_encoded['conso_clients_RES'].fillna(value=X_encoded['conso_totale'] * X_encoded['ratio_conso_RES'])
    X_encoded['conso_clients_PRO'] = X_encoded['conso_clients_PRO'].fillna(value=X_encoded['conso_totale'] * X_encoded['ratio_conso_PRO'])
    X_encoded['conso_clients_ENT'] = X_encoded['conso_clients_ENT'].fillna(value=X_encoded['conso_totale'] * X_encoded['ratio_conso_ENT'])
    X_encoded['prod_reseau_BT'] = X_encoded['prod_reseau_BT'].fillna(value=X_encoded['prod_totale'] * X_encoded['ratio_prod_BT'])
    X_encoded['prod_reseau_HTA'] = X_encoded['prod_reseau_HTA'].fillna(value=X_encoded['prod_totale'] * X_encoded['ratio_prod_HTA'])
    X_encoded['prod_filiere_eolien'] = X_encoded['prod_filiere_eolien'].fillna(value=X_encoded['prod_totale'] * X_encoded['ratio_prod_eolien'])
    X_encoded['prod_filiere_PV'] = X_encoded['prod_filiere_PV'].fillna(value=X_encoded['prod_totale'] * X_encoded['ratio_prod_PV'])
    X_encoded['prod_filiere_autre'] = X_encoded['prod_filiere_autre'].fillna(value=X_encoded['prod_totale'] * X_encoded['ratio_prod_autre'])

    # pour le territoire , on met 0 là ou c'est null
    X_encoded['type_territoire'] = X_encoded['type_territoire'].fillna(value=0)
 
    # On retourne en supprimant toutes les colonnes qui ne servent plus

    return X_encoded.drop(columns=["ratio_conso_BT", "ratio_conso_HTA", "ratio_prod_BT",
                              "ratio_prod_HTA", "ratio_conso_RES", "ratio_conso_PRO",
                               "ratio_conso_ENT", "ratio_prod_eolien", "ratio_prod_PV","ratio_prod_autre"])

    #return X_encoded



valnulles_encoder = FunctionTransformer(_encode_valnulles)


x_train = valnulles_encoder.fit_transform(x_train)

#%%

#%% traitement des données nulles


x_train.isnull().sum(axis = 0)


#%%
# on étudie les types de territoires
x_train['type_territoire'].value_counts()


#%% on essaie dateencoder sur x_train

#%%
# on voit qu'il y a des valeurs nulles dans les colonnes suivantes :
# conso_reseau_BT, conso_reseau_HTA, prod_reseau_BT, prod_reseau_HTA, conso_clients_RES, conso_clients_PRO, conso_clients_ENT
# compbien il ya d'observations ? 
x_train.shape
# ressort les lignes qui n'ont pas de conso_reseau_BT
x_train[x_train['conso_reseau_BT'].isnull()]



x_train.info()

#%%

# preparartion des données
# à faire :
# encoder les valeurs de la colonne "type_territoire" de x_train
# encoder les années et mois de la colonne "mois" de x_train
# faire les colonnes croisées prod x prop
# ajouter les polynomiales sur les prods et consos
# partir a priori sur une regression linéaire, ca semblerait assez logique

#%%



def _encode_dates(X):
    # on garde le mois et l'année pour l'instant
    # a priori sans impact sur les pertes techniques mais peut etre sur les pertes non techniques
    # exemple : plus de fraude l'hiver?
    # ne n_days pour voir s'il y a une évolution temporelle de la formule (amélioration du réseau ? de la detection de fraude?)
    X_encoded = X.copy()
    # Make sure that DateOfDeparture is of datetime format
    X_encoded['moisannee'] = pd.to_datetime(X_encoded['mois'], format="%m/%Y")
    # Encode the DateOfDeparture
    X_encoded['year'] = X_encoded['moisannee'].dt.year
    X_encoded['month'] = X_encoded['moisannee'].dt.month
    X_encoded['n_days'] = X_encoded['moisannee'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # la saison du mois
    X_encoded['saison'] = X_encoded['month'].apply(
        lambda date: 1 if date in [12, 1, 2] else 2 if date in [3, 4, 5] else 3 if date in [6, 7, 8] else 4
    )
    # Once we did the encoding, we will not need DateOfDeparture
    return X_encoded.drop(columns=["moisannee", "mois"])


date_encoder = FunctionTransformer(_encode_dates)


x_train = date_encoder.fit_transform(x_train)

#%%
x_train.head(100)

# il faut vérivier dans xtrain que la somme de conso_reseau_BT et conso_reseau_HT est égale à conso_totale
# idem pour prod_totale, prod_reseau_BT et prod_reseau_HT

x_train['conso_totale'].head(100)
constot = x_train['conso_reseau_BT']+x_train['conso_reseau_HTA']   # on voit que c'est égal à conso_totale
constot.head(100)

# on somme conso_clients_RES, conso_clients_PRO et conso_clients_ENT pour voir si c'est égal à conso_reseau_BT

consores = x_train['conso_clients_RES']+x_train['conso_clients_PRO']+x_train['conso_clients_ENT']
consores.head(100)

# utiliser ces inegalités pour combler les données manquantes
# puis supprimer les dimensions corréllés
#%%
x_train.info()

#%%
def _encode_prop(X):
    X_encoded = X.copy()
    # conso_totale x prop_conso_jour
    X_encoded['conso_totale_jour'] = X_encoded['conso_totale'] * X_encoded['prop_conso_jour']
    # prod_totale x prop_prod_jour
    X_encoded['prod_totale_jour'] = X_encoded['prod_totale'] * X_encoded['prop_prod_jour']
    # Prop_clts_logement_indiv x conso_clients_RES
    X_encoded['conso_clients_RES_logement_indiv'] = X_encoded['conso_clients_RES'] * X_encoded['prop_clts_logement_indiv']
    # Prop_clts_logement_collectif x conso_clients_RES
    X_encoded['conso_clients_RES_logement_collectif'] = X_encoded['conso_clients_RES'] * X_encoded['prop_clts_logement_collectif']
    # Prop_hta_type_1 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_1'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_1']
    # Prop_hta_type_2 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_2'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_2']
    # Prop_hta_type_3 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_3'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_3']
    # Prop_hta_type_4 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_4'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_4']
    
    # proposition de raamener tout ca à une prod et une conso par jour ... en divisant par le nombre de jours dans le mois, voire trouver un moyen pour intégrer les jours feriés
    
    # on supprime les colonnes qui ne servent plus
    return X_encoded.drop(columns=["prop_conso_jour", "prop_prod_jour", "prop_clts_logement_indiv", "prop_clts_logement_collectif", 
                                   "prop_hta_type_1", "prop_hta_type_2", "prop_hta_type_3", "prop_hta_type_4"])


prop_encoder = FunctionTransformer(_encode_prop)

x_train = prop_encoder.fit_transform(x_train)


#%%



#%% 

# on va ajouter les colonnes pour les puissances au carré

def _encode_puissance(X):
    X_encoded = X.copy()

    # puissance_transfos au carré
    X_encoded['puissance_transfos_2'] = X_encoded['puissance_transfos']**2
    # conso_totale au carré
    X_encoded['conso_totale_2'] = X_encoded['conso_totale']**2
    # prod_totale au carré
    X_encoded['prod_totale_2'] = X_encoded['prod_totale']**2
    # prod_reseau_BT au carré
    X_encoded['prod_reseau_BT_2'] = X_encoded['prod_reseau_BT']**2
    # prod_reseau_HTA au carré
    X_encoded['prod_reseau_HTA_2'] = X_encoded['prod_reseau_HTA']**2
    # prod_filiere_eolien au carré
    X_encoded['prod_filiere_eolien_2'] = X_encoded['prod_filiere_eolien']**2
    # prod_filiere_PV au carré
    X_encoded['prod_filiere_PV_2'] = X_encoded['prod_filiere_PV']**2
    # prod_filiere_autre au carré
    X_encoded['prod_filiere_autre_2'] = X_encoded['prod_filiere_autre']**2
    # conso_clients_RES au carré
    X_encoded['conso_clients_RES_2'] = X_encoded['conso_clients_RES']**2
    # conso_clients_PRO au carré
    X_encoded['conso_clients_PRO_2'] = X_encoded['conso_clients_PRO']**2
    # conso_clients_ENT au carré
    X_encoded['conso_clients_ENT_2'] = X_encoded['conso_clients_ENT']**2
    # conso_reseau_BT au carré
    X_encoded['conso_reseau_BT_2'] = X_encoded['conso_reseau_BT']**2
    # conso_reseau_HTA au carré
    X_encoded['conso_reseau_HTA_2'] = X_encoded['conso_reseau_HTA']**2

    return X_encoded

puissance_encoder = FunctionTransformer(_encode_puissance)

x_train = puissance_encoder.fit_transform(x_train)


#%%
    
x_train.info()

# on etudie les valeurs vides

#%% on compte les valeurs nulles par colonne
x_train.isnull().sum(axis = 0)
# on voit qu'il y a des valeurs nulles dans les colonnes suivantes :
# conso_reseau_BT, conso_reseau_HTA, prod_reseau_BT, prod_reseau_HTA, conso_clients_RES, conso_clients_PRO, conso_clients_ENT
# compbien il ya d'observations ? 
x_train.shape
# ressort les lignes qui n'ont pas de conso_reseau_BT
x_train[x_train['conso_reseau_BT'].isnull()]

#%%


# on transforme pour les  encoding de colonnes

def _encode_columns(X):

    # on encode les valeurs de la colonne "type_territoire" de x_train
    X_encoded = X.copy()

    return X_encoded.drop(columns=["type_territoire",
                                "id_poste_source", "id_dr",
                                "year", "month", "saison"])

#%%
columns_encoder = FunctionTransformer(_encode_columns)

x_train = columns_encoder.fit_transform(x_train)

#%%

x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)

#%%
x_train.info()

#%%


models = {
    "Random Forest": RandomForestRegressor(
        min_samples_leaf=5, random_state=0
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        learning_rate = 0.1, max_depth= 5, min_samples_leaf= 2, min_samples_split= 5, n_estimators= 157, subsample= 0.8, random_state=0, loss='squared_error'),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        max_iter = 80, max_depth= 5, min_samples_leaf= 3, learning_rate=0.1 , random_state=0, loss='squared_error'),
    "linearregression": LinearRegression(),
    "lasso": Lasso(alpha=0.5),
    "lassoCv": LassoCV(alphas=[10000, 100000, 10000000 ], max_iter=10, tol=0.01, cv = 5),
}


#%% MAintenant il faut faire une normalisation des données
# on va utiliser le standard scaler


#%%

x_train.shape

#%%

x_train.columns

#%%


path_to_data = "data"
# Load files
x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)
y_train = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)
x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)


def get_estimator():

    numeric_features = ['conso_totale', 'prod_totale',
       'conso_reseau_BT', 'conso_reseau_HTA', 'conso_clients_RES',
       'conso_clients_PRO', 'conso_clients_ENT', 'prod_reseau_BT',
       'prod_reseau_HTA', 'prod_filiere_eolien', 'prod_filiere_PV',
       'prod_filiere_autre', 'taux_linky', 'temperature',
       'long_reseau_aerien_bt', 'long_reseau_souterrain_bt', 'nb_postes_htabt',
       'puissance_transfos', 'n_days', 'conso_totale_jour', 'prod_totale_jour',
       'conso_clients_RES_logement_indiv',
       'conso_clients_RES_logement_collectif', 'prod_reseau_HTA_type_1',
       'prod_reseau_HTA_type_2', 'prod_reseau_HTA_type_3',
       'prod_reseau_HTA_type_4', 'puissance_transfos_2', 'conso_totale_2',
       'prod_totale_2', 'prod_reseau_BT_2', 'prod_reseau_HTA_2',
       'prod_filiere_eolien_2', 'prod_filiere_PV_2', 'prod_filiere_autre_2',
       'conso_clients_RES_2', 'conso_clients_PRO_2', 'conso_clients_ENT_2',
       'conso_reseau_BT_2', 'conso_reseau_HTA_2']
    # Création d'un preprocessor pour le ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ], remainder='passthrough')

    pileline = Pipeline([
        ('null',valnulles_encoder), 
        ('date',date_encoder),
        ('prop', prop_encoder), 
        ('poly',puissance_encoder), 
        ('col',columns_encoder),
        ('prep',preprocessor),
        ('mod',models["lassoCv"])]
    )

    return pileline



model = get_estimator()

model.fit(x_train, y_train)

#%%

y_pred = model.predict(x_test)

#%%
# Missing values
# on remplace les valeurs manquantes par la moyenne pour les seules colonnes numériques de
# x_train

x_train.info()
#%%
#%%
# on supprime les trois premiers colonnes de x_train
x_train = x_train.iloc[:, 3:]
#%%

x_train = x_train.apply(lambda col: col.fillna(col.mean()) if col.dtype.kind == 'f' else col, axis=0)

#%% compte les valeures nulles par colonne
x_train.isnull().sum(axis = 0)

#%%
#x_train.fillna(x_train.mean(), inplace=True)

#%%

# Fit model
model = RandomForestRegressor(n_estimators = 10, random_state=0, max_depth=5, criterion="absolute_error")
model.fit(x_train, y_train.values.ravel())

#%%
# Prediction

#%%

# on supprime les 3 premieres colonnes de x_test
x_test = x_test.iloc[:, 3:]

# on encode les valeurs de la colonne "type_territoire" de x_test
x_test["type_territoire"] = le.fit_transform(x_test["type_territoire"])
x_test.fillna(x_test.mean(), inplace=True)

#%%
predicted_values = model.predict(x_test)

#%%

y_pred = pd.DataFrame(data=predicted_values, index=x_test.index.values, columns=["pertes_totales"])
y_pred.index.name = 'ID'

y_pred.to_csv(path_to_data + "/testing_benchmark2.csv")

#%%
# on calcule l'erreur sur le jeu d'entrainement
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# on prend l'erreur absolue moyenne

y_pred_train = model.predict(x_train)
mae = mean_absolute_error(y_train, y_pred_train)
print("MAE : ", mae)


#%%

# on étudie les coefficients de la regression linéaire

lasso_coefficients = model.named_steps['mod'].coef_

#%%
# Afficher les coefficients
print('Coefficients du modèle LASSO :\n', lasso_coefficients)

#%%


# Récupérer les noms des caractéristiques après le prétraitement
columns_after_transform = model.named_steps['prep'].get_feature_names_out()

#%%
# Créer un DataFrame avec les noms des caractéristiques et les coefficients associés
coefficients_df = pd.DataFrame({'Feature': columns_after_transform, 'Coefficient': lasso_coefficients})

# Afficher les coefficients
print('Coefficients du modèle LASSO avec noms de caractéristiques :\n', coefficients_df)

#%%

alphas = model.named_steps['mod'].alpha_

alphas

#%%
plt.figure(figsize=(12, 8))

for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], label=f'Coefficient {i}')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Trajectoire des coefficients pour différentes valeurs d\'alpha')
plt.legend()
plt.show()