import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
import scipy
import seaborn as sns

dataset = pd.read_csv("../BD/dataframe_YesIndex_YesHeader_C.csv")

print("Hi han valors NaN a la base de dades? " + str(dataset.isnull().values.any()))

'''Adaptar les dades (normalitzar)'''
dataset1 = dataset.values
idsx = list(range(22))
idsx.pop(18)
dataset1 = dataset1[:,idsx]
scaler = MinMaxScaler()
scaler.fit(dataset1)
dataset1 = scaler.transform(dataset1)

idsx.pop(20)
y = dataset1[:,18]
x = dataset1[:,idsx]


'''Generar grafiques'''
for i in range(x.shape[1]):
        plt.scatter(x[:,i], y)
        plt.savefig("../Grafiques/disp/" + str(i) + ".png")
        plt.clf()
        density = scipy.stats.gaussian_kde(x[:,i])
        n, xi, _ = plt.hist(x[:,i], density=True)
        plt.plot(xi, density(xi))
        plt.savefig("../Grafiques/hist/" + str(i) + ".png")
        plt.clf()

