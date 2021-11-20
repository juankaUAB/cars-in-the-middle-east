import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import math


dataset = pd.read_csv("../BD/dataframe_YesIndex_YesHeader_C.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])
dataset = dataset.drop_duplicates()
dataset = dataset[["Torque","Cylinders","Horsepower","Top Speed","Engine Capacity","price"]]

dataset1 = dataset.values

x = dataset1[:,:5]
y = dataset1[:,5]

'''Dividim el conjunt en train i test'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


'''COMENÇEM LA CERCA EXHAUSTIVA DELS MILLORS PARAMETRES'''
models = [LinearRegression(), DecisionTreeRegressor()]
nombres_models = ["Linear Regression", "Decision Tree"]
parametres = [{'fit_intercept': [True, False], 'normalize': [True,False]},
              {"splitter":["best","random"],
            "max_depth" : [1,3,5,10,15,25,30,50],
            "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,50,80,100,250,500] }]



resum = []
for i, model in enumerate(models):
    print("----" + str(nombres_models[i]) + "----")
    print("")
    clf = GridSearchCV(estimator=model, param_grid=parametres[i], cv=3, verbose=3, n_jobs=-1)
    resum.append(clf.fit(X_train, y_train))
    print("Els millors parametres" + str(resum[i].best_params_))
    print("La millor score: " + str(resum[i].best_score_))
    print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,clf.predict(X_test)))))
    print("")
    
    
'''CERCA EXHAUSTIVA PER EL RANDOM FOREST'''

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print("----Random Forest----")
random_tree = RandomForestRegressor()
randomCV = RandomizedSearchCV(estimator = random_tree, param_distributions = random_grid, 
                              n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
randomCV.fit(X_train, y_train)
print("Els millors parametres: " + str(randomCV.best_params_))
print("La millor score: " + str(randomCV.best_score_))
print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,randomCV.predict(X_test)))))
print("")

