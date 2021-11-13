import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import scipy
import seaborn as sns
import graphviz

dataset = pd.read_csv("../BD/dataframe_YesIndex_YesHeader_C.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])
dataset = dataset.drop_duplicates()
dataset = dataset[["Torque","Cylinders","Horsepower","Top Speed","Engine Capacity","price"]]

'''Adaptar les dades (normalitzar)'''
dataset1 = dataset.values
scaler = MinMaxScaler()
scaler.fit(dataset1)
dataset1 = scaler.transform(dataset1)

x = dataset1[:,:5]
y = dataset1[:,5]

'''Dividim el conjunt en train i test'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

'''Creem el model de regressió lineal'''
print("----REGRESSIO LINEAL----")
model = LinearRegression()
model.fit(X_train, y_train)
prediccions = model.predict(X_test)

print("Score del model: " + str(model.score(X_test, y_test)))

'''Visualitzem la regressió'''
plt.figure()
ax = plt.scatter(X_test[:,0], y_test)
plt.plot(X_test, prediccions, 'r')
plt.savefig("../demo-linregr.png")
plt.clf()

'''Calculem el MSE'''
mse = mean_squared_error(y_test, prediccions)
print("Error quadratic mitja: " + str(mse))

'''Decision Tree Regressor'''
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print("---- DECISION TREE----")
print("Profunditat del arbre: " + str(decision_tree.get_depth()))
print("Numero de fulles del arbre: " + str(decision_tree.get_n_leaves()))
prediccions_arbre = decision_tree.predict(X_test)

print("Score del model: " + str(decision_tree.score(X_test, y_test)))

'''Calculem el MSE'''
mse = mean_squared_error(y_test, prediccions_arbre)
print("Error quadratic mitja: " + str(mse))

'''Visualitzem els resultats'''
ax = plt.scatter(X_test[:,0], y_test)
plt.plot(X_test, prediccions_arbre, 'r')
plt.savefig("../demo-decisiontree.png")
plt.clf()

dot_data = export_graphviz(decision_tree, out_file='../decision_tree.dot')

'''Random Forest Regressor'''
print("---- RANDOM FOREST REGRESSOR ----")
random_tree = RandomForestRegressor()
random_tree.fit(X_train, y_train)

print("Score del model: " + str(random_tree.score(X_test, y_test)))
prediccions_randtree = random_tree.predict(X_test)
mse = mean_squared_error(y_test, prediccions_randtree) 
print("Error quadratic mitja: " + str(mse))

'''Visualitzem els resultats'''
ax = plt.scatter(X_test[:,0], y_test)
plt.plot(X_test, prediccions_randtree, 'r')
plt.savefig("../demo-randomtree.png")
plt.clf()


