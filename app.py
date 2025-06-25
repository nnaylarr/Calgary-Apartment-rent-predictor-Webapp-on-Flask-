#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:44:47 2023

@author: syedahmadsohail
"""

# Importing required libraries
import numpy as np
import pandas as pd
import pickle
import sklearn

# Loading the dataset
dataset = pd.read_csv('rent.csv')

# Filling missing values in 'bed_room' with 0
dataset['bed_room'].fillna(0, inplace=True)

# Filling missing values in 'area' with the mean
dataset['area'].fillna(dataset['area'].mean(), inplace=True)

# Converting neighborhood names into numeric codes
neighborhood_dict = {'NE': 4, 'NW': 5, 'SE': 6, 'SW': 7}
dataset['location'] = dataset['location'].apply(lambda x: neighborhood_dict[x])

# Selecting features
X = dataset.iloc[:, :3]

# Function to convert bedroom text to integer
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'studio': 0}
    return word_dict[word]

# Applying the conversion to the 'bed_room' column
X['bed_room'] = X['bed_room'].apply(lambda x: convert_to_int(x))

# Selecting the target variable
y = dataset.iloc[:, -1]

# Training the Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Saving the trained model to a file
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading the model back from file
model = pickle.load(open('model.pkl', 'rb'))

# Sample input for prediction
input_data = [2, 2200, 5]  # Format: [bed_room, area, location]

# Simple input validation
if not all(isinstance(i, (int, float)) for i in input_data):
    print("Invalid input. All values must be numeric.")
else:
    prediction = model.predict([input_data])[0]
    print(f"Predicted rent for inputs {input_data} is: ${round(prediction, 2)}")
