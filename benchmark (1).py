#%%

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

#%%

path_to_data = "data"
# Load files
x_train = pd.read_csv(path_to_data + "/training_input.csv", index_col=0)

#%%
y_train = pd.read_csv(path_to_data + "/training_output.csv", index_col=0)

#%%
x_test = pd.read_csv(path_to_data + "/testing_input.csv", index_col=0)

#%%

x_train

#%%
x_test

#%%
# on exporte x_train dans un fichier
x_train.to_csv("x_train.csv")

#%%

# Encoding
le = LabelEncoder()
# on veut encoder les valeurs de la colonne "type_territoire" de x_train

x_train["type_territoire"] = le.fit_transform(x_train["type_territoire"])
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

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Récupération de l'importance des features
feature_importances = model.feature_importances_

# Création d'un DataFrame pour afficher les importances
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances})

# Trier le DataFrame par importance descendante
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Affichage des 10 premières features les plus importantes
print(feature_importance_df.head(10))

# Tracé d'un graphique des importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 des importances des features dans le modèle RandomForest')
plt.show()