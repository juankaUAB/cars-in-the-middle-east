import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy
import seaborn as sns

dataset = pd.read_csv("../BD/dataframe_YesIndex_YesHeader_C.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])
dataset = dataset.drop_duplicates()
dataset = dataset[["Engine Capacity","Cylinders","Horsepower","Torque","Top Speed","price"]]

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
model = LinearRegression()
model.fit(X_train, y_train)
prediccions = model.predict(X_test)

precisio = model.score(X_train, y_train)
print(precisio)

'''Visualitzem la regressió'''
plt.figure()
ax = plt.scatter(X_test[:,0], y_test)
plt.plot(X_test, prediccions, 'r')
plt.savefig("../demo-linregr.png")


