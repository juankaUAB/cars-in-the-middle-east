import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy
import seaborn as sns

titols = ["Engine Capacity","Cylinders","Drive Type","Tank Capacity", "Fuel Economy","Fuel Type","Horsepower","Torque","Transmission","Top Speed","Seating Capacity","Acceleration","Length","Width","Height","Wheelbase","Trunk Capacity","Currency","Country"]
equivalencies = {'0': 0.23, '1': 0.24, '5': 2.32, '3': 2.89, '4': 2.27, '2': 0.24}
dataset = pd.read_csv("../BD/dataframe_YesIndex_YesHeader_C.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])
dataset = dataset.drop_duplicates()

dataset1 = pd.read_csv("../BD/Original_raw_YesIndex_YesHeader.csv")

'''Prova de que el mateix cotxe existeix en paisos diferents'''
print(dataset[dataset["name"] == "Mitsubishi Attrage 2021 1.2 GLX (Base)"])

print("Hi han valors NaN a la base de dades? " + str(dataset.isnull().values.any()))

'''Adaptar les dades (normalitzar)'''
dataset1 = dataset.values
idsx = list(range(21))
idsx.pop(17)
dataset1 = dataset1[:,idsx]
scaler = MinMaxScaler()
scaler.fit(dataset1)
dataset1 = scaler.transform(dataset1)

idsx.pop(19)
y = dataset1[:,17]
x = dataset1[:,idsx]


'''Generar grafiques'''
for i in range(x.shape[1]):
        if i != 19:
            plt.xlabel(titols[i])
            plt.ylabel("Price")
            plt.scatter(x[:,i], y)
            plt.savefig("../Grafiques/disp/" + str(i) + ".png")
            plt.clf()
            density = scipy.stats.gaussian_kde(x[:,i])
            n, xi, _ = plt.hist(x[:,i], density=True)
            plt.plot(xi, density(xi))
            plt.savefig("../Grafiques/hist/" + str(i) + ".png")
            plt.clf()

fig, ax = plt.subplots(figsize=(20,20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(dataset.corr(), ax=ax, cmap=cmap, vmin=0, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig("../Grafiques/heatmap/mapa-calor.png")

'''Calculem la desviació estandar de cada atribut'''
desviacions = np.std(dataset1,axis=0)
with open("../Estadistiques/desviacions.txt",'w') as d:
    for i, des in enumerate(desviacions):
        d.write("Atributo " + str(i+1) + " : " + str(des) + "\n")
        d.write("----------------------------\n")
        

'''Apliquem un test de normalitat (el de Shapiro) a cadascuna de les variables per determinar 
quines ens seran utils (segueixen una distribuicio normal)'''
resultats = []
for i in range(x.shape[1]):
    resultats.append(scipy.stats.shapiro(x[:,i]))
resultats = np.array(resultats)
    
with open("../Estadistiques/testNormalitat.txt",'w') as f:
    f.write(" - TEST DE SHAPIRO - \n")
    f.write("---------------------\n")
    for k, res in enumerate(resultats):
        f.write("Atributo " + str(k+1) + " : Estadistico: " + str(res[0]) + "   |   P-Valor: " + str(res[1]) + "\n")
        if res[1] < 0.05:
            f.write("Se puede rechazar la hipotesis de que los datos de distribuyen de forma normal\n")
        else:
            f.write("No se puede rechazar la hipotesis de que los datos de distribuyen de forma normal\n")
        f.write("-----------------------------------------------------------------------------\n")
        
'''Diagrama de barres'''
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.barplot(x="Country", y="price", data=dataset[["price","Country"]],palette='pastel')
plt.savefig("../Grafiques/bar/diagrama-barres.png")

