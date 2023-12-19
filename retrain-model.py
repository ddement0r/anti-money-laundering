# %%
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle


# %%
#Importing the dataset
df = pd.read_csv('mldataset2.csv')

# %%
#printing the dataset
df

# %%
#Printing information about the dataset
df.info()

# %%
#deleting useless columns (step, nameOrig, nameDest, isFlaggedFraud)
df = df.drop(['step', 'isFlaggedFraud','nameOrig','nameDest'], axis=1)

# %%
df

# %%
#the rows with newbalanceDest are the same with nameDest = 'M*******', the dataset owner claims that there are no information for Merchants
df = df[df['newbalanceDest'] != 0]

# %%
#Printing the dataset (the number of rowns is reduced from 6362620 rows to 3923187 rows)
df

# %%
#checking if there are Null Values
df.isnull().sum()

# %%
#checking if there are duplicated values
df.duplicated().sum

# %%
#Replacing the categorical Feature 'type' with the integers 1, 2, 3 and 4
df = df.replace('DEBIT',1)
df = df.replace('CASH_OUT',2)
df = df.replace('TRANSFER',3)
df = df.replace('CASH_IN',4)

# %%
df

# %%
#To make sure all the 'type' values are integers 
df.info()

# %%
df.value_counts('isFraud')

# %%
df.corr()

# %%
sns.heatmap(df.corr(), annot=True)
plt.show()

# %%
X = df.drop(["isFraud"], axis=1)
y = df["isFraud"]

# %%
print(X)

# %%
print(y)

# %%
df

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# %%
rf_model = RandomForestClassifier(n_estimators=50, random_state=44)

# %%
rf_model.fit(X_train, y_train)

# %%
predictions = rf_model.predict(X_test)

# %%
model_file = "antilaundering.pkl"

# %%
with open(model_file, 'wb') as file:  
    pickle.dump(rf_model, file)

# %%
rf_model


