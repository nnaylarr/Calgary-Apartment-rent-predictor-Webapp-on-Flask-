#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:44:47 2023

@author: syedahmadsohail
"""

# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import sklearn

dataset = pd.read_csv('rent.csv')

dataset['bed_room'].fillna(0, inplace=True)

dataset['area'].fillna(dataset['area'].mean(), inplace=True)
# convert neighborhood to numeric values
neighborhood_dict = {'NE': 4, 'NW': 5, 'SE': 6, 'SW': 7}
dataset['location'] = dataset['location'].apply(lambda x: neighborhood_dict[x])
X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'studio':0}
    return word_dict[word]

X['bed_room'] = X['bed_room'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))