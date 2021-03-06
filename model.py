
# import independnecies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# read csv with pandas
df = pd.read_csv('cleaned.csv')

# cleaning process
df = df.drop(columns=['Week Start','Cases - Cumulative','Percent Tested Positive - Weekly','Deaths - Cumulative','Death Rate - Cumulative',
       'Population','Percent Tested Positive - Cumulative','Test Rate - Weekly','Tests - Cumulative','Test Rate - Cumulative','Case Rate - Cumulative','Week End','Week Number','Test Rate - Cumulative','Row ID','ZIP Code Location', 'ZIP Code'])

# Fix Column Names
df.columns = df.columns.str.strip().str.lower().str.replace('-', '').str.replace(' ', '_').str.replace('__', '_')


# create model variables
X = df.drop('deaths_weekly', axis = 1)
y = df['deaths_weekly'].values.reshape(-1, 1)


# split train_test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# initiate a linear regression model and train data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#save model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))