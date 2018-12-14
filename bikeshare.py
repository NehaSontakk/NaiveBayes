#Trip history analysis of bike data
#split into training and test and use NB classifier
#Dataset from: https://www.capitalbikeshare.com/system-data

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

data=pd.read_csv('/home/ubuntu/Assignments/LP-1/DA/A4/201808-capitalbikeshare-tripdata.csv')

print(data.head())
print(data.isnull().sum())

#Drop data columns that are not needed
data=data.drop('Start date',axis=1)
data=data.drop('End date',axis=1)
data=data.drop('Start station',axis=1)
data=data.drop('End station',axis=1)

#Label Encoder to encode all our unique non-numerical values with unique numbers.
data.head()
le = LabelEncoder()
le.fit(data['Member type'])
data['Member type'] = le.transform(data['Member type'])

le = LabelEncoder()
le.fit(data['Bike number'])
data['Bike number'] = le.transform(data['Bike number'])

print(data.head())
print(data.shape)

train=np.array(data.iloc[0:85000])
test=np.array(data.iloc[85000:,])

print(train.shape,test.shape)

#Gaussian NB assumes features follow normal distribution and is better for categorical values.
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(train[:,0:4],train[:,4])
predicted=model.predict(test[:,0:4])

print(predicted.shape)

count=0.0
for l in range(30597):
    if(predicted[l]==test[l,4]):
        count=count+1
print(count)

#Accuracy Percentage
print(count/30597)
