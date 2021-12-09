# Practica Kaggle APC UAB 2021-22
### Name: Juan Carlos Martínez Moreno  
### DATASET: Cars in the Middle East  
### URL: [kaggle](https://www.kaggle.com/bushnag/cars-in-the-middle-east)  
![Cars](cotxe.jpg)

## Summary  
Our dataset contains information about aprox. 4000 cars from the Middle East. In the dataset there are cars from 6 differents countries: Saudi Arabia, USA, Kuwait, Bahrain, Oman and Qatar.  
We have 4000 rows and 21 columns. The columns are the attributes of the dataset that we will analyse in order to select the best attributes to do the prediction, and they are the features of a car, like the torque, Engine Capacity, Cylinders, Fuel Economy, size of the car in length, width and height, etc...  

## Objectives
The objectives that we want to achieve are to predict prices of some cars in different countries since their features, and see in which country is better to buy a specific car.

## Preprocessing
To prepare the data for the prediction task, it is compulsory to analyse the attributes of the dataset and choose the best that will do the best regression. To do that, there are many possibilities: histograms, scatter plots, heatmap (correlation of attributes), apply normal test to see the distribution of all attributes (interesting to choose those who follow normal distribution), and we can normalize data if the attributes have diferent ranges.  
In our dataset, for example, cylinders and size of car do not have same ranges.

## Models
| Model | Hyperparameters  | R2 Score  | MSE |
|---|---|---|---|
| Multivariable Linear Regressor  | {'fit_intercept': True}  | 0.24 | 0.07  |
| Decision Tree Regressor  | {'max_depth': 15, 'max_features': 'auto', 'max_leaf_nodes': 20, 'min_weight_fraction_leaf': 0.0, 'splitter': 'random'}  |  0.61 |  0.06  |
| Random Forest  | {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False}  | 0.58  | 0.06  |

## Demo
To test the code, you can clone my github repository by executing this in cmd:  
```git clone https://github.com/juankaUAB/cars-in-the-middle-east.git```  
You will need some libraries to test my code, so you can install them executing this:  
```pip install -r requirements.txt```

## Conclusions
The prediction in the dataset is difficult because every country has it own currency, so the models do not have information about where the car is from. However, Decision Tree and Random Forest have good results, and we can use them in this problem.
