
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

    return X_encoded



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
    """ X_encoded['n_days'] = X_encoded['moisannee'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # la saison du mois
    X_encoded['saison'] = X_encoded['month'].apply(
        lambda date: 1 if date in [12, 1, 2] else 2 if date in [3, 4, 5] else 3 if date in [6, 7, 8] else 4
    ) """
    # Once we did the encoding, we will not need DateOfDeparture
    return X_encoded.drop(columns=["moisannee", "mois"])





#x_train = date_encoder.fit_transform(x_train)

def _encode_prop(X):
    X_encoded = X.copy()
    # conso_totale x prop_conso_jour
    X_encoded['conso_totale_jour'] = X_encoded['conso_totale'] * X_encoded['prop_conso_jour'] /100
    # prod_totale x prop_prod_jour
    X_encoded['prod_totale_jour'] = X_encoded['prod_totale'] * X_encoded['prop_prod_jour'] /100
    # Prop_clts_logement_indiv x conso_clients_RES
    X_encoded['conso_clients_RES_logement_indiv'] = X_encoded['conso_clients_RES'] * X_encoded['prop_clts_logement_indiv'] /100
    # Prop_clts_logement_collectif x conso_clients_RES
    X_encoded['conso_clients_RES_logement_collectif'] = X_encoded['conso_clients_RES'] * X_encoded['prop_clts_logement_collectif'] /100
    # Prop_hta_type_1 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_1'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_1'] /100
    # Prop_hta_type_2 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_2'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_2'] /100
    # Prop_hta_type_3 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_3'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_3'] /100
    # Prop_hta_type_4 x prod_reseau_HTA
    X_encoded['prod_reseau_HTA_type_4'] = X_encoded['prod_reseau_HTA'] * X_encoded['prop_hta_type_4'] /100
    # linky x conso_totale
    X_encoded['conso_linky'] = X_encoded['conso_totale'] * X_encoded['taux_linky'] /100
    # divers
    X_encoded['con_prod_hors_linky'] = (X_encoded['conso_totale']-X_encoded['prod_totale']) * (100-X_encoded['taux_linky']) /100
    # divers
    X_encoded['con_prod_BT'] = X_encoded['conso_reseau_BT'] - X_encoded['prod_reseau_BT']

    # proposition de raamener tout ca à une prod et une conso par jour ... en divisant par le nombre de jours dans le mois, voire trouver un moyen pour intégrer les jours feriés
    
    # on supprime les colonnes qui ne servent plus
    return X_encoded


# x_train = prop_encoder.fit_transform(x_train)

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
    # conso_RES_linky au carré
    X_encoded['conso_linky_2'] = X_encoded['conso_linky']**2
    # Divers
    X_encoded['con_prod_hors_linky_2'] = X_encoded['con_prod_hors_linky']**2
    # Divers
    X_encoded['con_prod_BT_2'] = X_encoded['con_prod_BT']**2
    return X_encoded


# x_train = puissance_encoder.fit_transform(x_train)

def _purge_data(X):
    X_encoded = X.copy()

    col_doublon = ['conso_reseau_HTA', 'conso_clients_PRO', 'prod_reseau_HTA', 'prod_filiere_autre', 'prod_reseau_HTA_type_4']

    col_correl = ['conso_totale_jour', 'prod_totale', 'conso_clients_ENT']

    col_inut = ['prod_totale_2', 'prod_reseau_HTA', 'prod_reseau_HTA_2', 'prod_filiere_eolien_2',
            'prod_filiere_eolien', 'prod_reseau_HTA_type_1', 'prod_reseau_HTA_type_4', 
            'saison', 'type_territoire', 'year',
            #on garde 'month' pour l'instant
            'conso_totale_2', 
            'prod_reseau_HTA_type_3', 'conso_clients_ENT_2',
            'prod_reseau_HTA_type_2', 'prod_filiere_autre_2', 'prod_filiere_PV_2']
    
    col_lasso = ['prod_filiere_eolien_2',
            'puissance_transfos_2',
            'conso_totale_2',
            'prod_totale_2',
            'conso_clients_ENT',
            'prod_reseau_HTA_2',
            'conso_clients_RES',
            'prod_filiere_PV_2',
            'conso_clients_PRO',
            'conso_linky',
            'conso_reseau_HTA',
            'conso_clients_ENT_2',
            'conso_reseau_BT_2',
            'conso_reseau_HTA_2',
            'prod_reseau_BT',
            'prod_reseau_HTA_type_2',
            'prod_reseau_HTA_type_4',
            'prod_reseau_HTA_type_3',
            'prop_hta_type_2',
            'prod_reseau_HTA_type_1',
            'prod_reseau_HTA',
            'prod_filiere_eolien',
            'conso_totale_jour',
            'ratio_prod_autre',
            'prod_filiere_autre',
            'ratio_prod_eolien',
            'ratio_conso_PRO',
            'prod_totale',
            'ratio_prod_HTA',
            'ratio_conso_HTA',
            'prop_clts_logement_collectif',
            'conso_totale']
    
    col_tech= ["ratio_conso_BT", "ratio_conso_HTA", "ratio_prod_BT",
            "ratio_prod_HTA", "ratio_conso_RES", "ratio_conso_PRO",
            "ratio_conso_ENT", "ratio_prod_eolien", "ratio_prod_PV","ratio_prod_autre"]

    col_init = ["prop_conso_jour", "prop_prod_jour", "prop_clts_logement_indiv", "prop_clts_logement_collectif", 
            "prop_hta_type_1", "prop_hta_type_2", "prop_hta_type_3", "prop_hta_type_4"]
    
    col_purge = ['conso_clients_RES_2',
            'ratio_prod_PV',
            'conso_clients_PRO_2',
            'temperature',
            'prop_clts_logement_indiv',
            'prod_filiere_autre_2',
            'prop_conso_jour',
            'conso_clients_RES_logement_indiv',
            'ratio_conso_RES',
            'prod_reseau_BT_2',
            'ratio_conso_BT',
            'conso_linky_2',
            'prop_hta_type_4',
            'prod_filiere_PV',
            'prop_hta_type_1',
            'ratio_prod_BT',
            'prop_hta_type_3',
            'ratio_conso_ENT', 
            'prop_prod_jour',
            'puissance_transfos',
            'long_reseau_aerien_bt',
            'con_prod_BT',
            'con_prod_BT_2']
    col =  col_lasso + col_purge # + col_tech # col_doublon + col_correl #+ col_inut

    # on veut la liste de toutes les colonnes de X_encoded
    #col_all = list(X_encoded.columns)
    # on veut retirer conso_reseau_BT de la liste coll_all
    #col_all.remove('conso_reseau_BT')
    #col = col_all

    # on enlève les noms en double dans col
    col = list(dict.fromkeys(col))

    return X_encoded.drop(columns=col)



valnulles_encoder = FunctionTransformer(_encode_valnulles)
date_encoder = FunctionTransformer(_encode_dates)
prop_encoder = FunctionTransformer(_encode_prop)
puissance_encoder = FunctionTransformer(_encode_puissance)
purge_data = FunctionTransformer(_purge_data)


