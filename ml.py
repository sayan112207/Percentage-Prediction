#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt  

# Load the dataset
link = "http://bit.ly/w-data"
data = pd.read_csv(link)

#Printing the loaded Dataset
data

data.plot(x='Hours', y='Scores', style='x')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage')  
plt.show()

# Extract the features and target variable
X = data['Hours'].values.reshape(-1, 1)
y = data['Scores'].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model and train it on the training data
model = LinearRegression()
model.fit(X_train, y_train)

import pickle
pickle.dump(model,open('model.pkl','wb'))

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)