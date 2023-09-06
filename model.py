import pandas as pd
from google.colab import drive
from matplotlib import pyplot as plt
drive.mount('/content/drive')
import numpy as np
import seaborn as sns

# Loading dataset from google drive 
dataset = '/content/drive/MyDrive/customer_churn_large_dataset.csv'
df = pd.read_csv(dataset)

# Checking for missing values
print(df.isnull().sum())

# the distribution of numerical features
sns.histplot(df['Age'], bins=20)
plt.title('Age Distribution')
plt.show()

#the distribution of categorical features
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.show()


#the relationship between features and churn
sns.boxplot(data=df, x='Churn', y='Monthly_Bill')
plt.title('Monthly Bill vs Churn')
plt.show()

#Feature anyls
# statistics for churned and non-churned customers
churn_stats = df.groupby('Churn').agg({'Monthly_Bill': ['mean', 'median', 'std']})
print(churn_stats)


#Creating a feature 'Age_Group' based on age ranges
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, float('inf')], labels=['<18', '18-35', '36-60', '60+'])

# Encoding categorical variables (e.g., one-hot encoding for 'Location' and 'Gender')
df = pd.get_dummies(df, columns=['Location', 'Gender'], drop_first=True)

# Encoding Age_group
df = pd.get_dummies(df, columns=['Age_Group'] , drop_first=True)

#Droping non using column
df = df.drop(['CustomerID' , 'Name' , 'Age'], axis=1)


#spliting the data into input and target
inputData = df.drop(['Churn'] , axis = 1)
targetVariable = df['Churn']


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputData, targetVariable, test_size=0.2, random_state=42)

# Initializing the Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Defining hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(10, 200),            # Number of trees in the forest
    'max_depth': randint(1, 20),                # Maximum depth of the tree
    'min_samples_split': randint(2, 20),       # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 20),        # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]                 # Whether to use bootstrap samples when building trees
}

# Initializing RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# Performing hyperparameter tuning with cross-validation
random_search.fit(X_train, y_train)

# Geting the best estimator from RandomizedSearchCV
best_model = random_search.best_estimator_

# Making predictions using the best model
y_pred = best_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classificationReport = classification_report(y_test, y_pred)

print(f'Best Model Parameters: {random_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_report}')

import joblib

# Saving the trained model to a file
filename = '/content/drive/MyDrive/mlModel'
joblib.dump(best_model, filename)
