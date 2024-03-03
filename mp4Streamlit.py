import streamlit as st
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

st.markdown("MINI PROJECT 4")
st.markdown("MACHINE LEARNING FOR ANALYSIS AND PREDICTION OF ATTRITION")
st.markdown("In this dataset we will look at the attrition of employees in a large company, let start by looking at a table of some all the data - notice how you in streamlit can click the collum headers and sort by... we only do this because we have such a small dataset")
# - Import all the needed dependecies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# - Load the dataset(csv) and show a sample of 6 rows

data = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv', delimiter=',')
data.sample(n=6)
st.write(data)
st.markdown("by clicking, we can see that we have a total of 1470 employees, and that the oldest is 60 - this is a fast way of looking at data, if you are a non coder.")
# - Show how many values are null
# - We have 0 null

data.isnull().sum()

st.markdown("Display the number of col and rows")
st.write(data.shape)

# - Lets print out some meaningfull statistical info about the data

data.describe()

# - We keep the Columns 'DistanceFromHome', 'HourlyRate' and 'Attrition' for further analysis towards attrition
# - And check again(how ever it is not needed) for null values

# +
import pandas as pd

# columns_to_keep is a list of column names we want to keep
columns_to_keep = ['DistanceFromHome', 'HourlyRate', 'Attrition']

columns_to_keep_all = ['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EmployeeNumber', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Using loc to select columns by label
df_new = data.loc[:, columns_to_keep]

df_new_all = data.loc[:, columns_to_keep_all]

# Using iloc to select columns by integer position
# You can use df.columns.get_loc() to get the integer position of column names
column_positions = [data.columns.get_loc(col) for col in columns_to_keep]
df_new = data.iloc[:, column_positions]
df_new.isnull().sum()

column_positions = [data.columns.get_loc(col) for col in columns_to_keep_all]
df_new_all = data.iloc[:, column_positions]
df_new_all.isnull().sum()
# -

# - This is the new data set 5 first columns

df_new.head()

# - We convert the 'Yes' or 'No' values to the numeric values 1 for 'Yes' and 0 for 'No'

# +
# Convert 'Attrition' column values in df_new DataFrame
df_new.loc[:, 'Attrition'] = df_new['Attrition'].map({'Yes': 1, 'No': 0}).astype('int64')

# Convert 'Attrition' column values in df_new_all DataFrame
df_new_all.loc[:, 'Attrition'] = df_new_all['Attrition'].map({'Yes': 1, 'No': 0}).astype('int64')

# Filter rows where 'Attrition' values are not 0 or 1
valid_values = df_new['Attrition'].isin([0, 1])
valid_values_all = df_new_all['Attrition'].isin([0, 1])


# Remove rows with invalid 'Attrition' values
df_new = df_new[valid_values]

# Remove rows with invalid 'Attrition' values
df_new_all = df_new_all[valid_values_all]


#Convert 'Attrition' column values to integers
df_new['Attrition'] = df_new['Attrition'].astype('int64')
df_new_all['Attrition'] = df_new_all['Attrition'].astype('int64')

print(df_new['Attrition'].unique())
print(df_new_all['Attrition'].unique())
# -

# Check the data type of the 'Attrition' column
attrition_dtype = df_new['Attrition'].dtype


df_new_all.dtypes

#howcome Attrition is still an object. let explore:
print(df_new['Attrition'].value_counts())
print(df_new['Attrition'].isnull().sum())
print(df_new_all['Attrition'].value_counts())
print(df_new_all['Attrition'].isnull().sum())
df_new.shape
df_new_all.shape

# # 2. Supervised machine learning: classification
# ## train, test, and validate two machine learning models for classification and prediction of attrition (e.g. Decision Tree and Naïve Bayes)
#
# ## apply appropriate methods and measures for assessing the validity of the models and recommend the
# one with highest accuracy
#
# - The result gives us Naïve Bayes selected as the best model.

# +
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




# Prepare the data
X = df_new.drop(columns=['Attrition'])  # Features
y = df_new['Attrition']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier model
dt_classifier = DecisionTreeClassifier()

# Train the Decision Tree classifier
dt_classifier.fit(X_train, y_train)

# Predict using the Decision Tree classifier
dt_predictions = dt_classifier.predict(X_test)

# Evaluate the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)



#lets see how this works out


# Train the Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Import necessary libraries
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score

# Predict using the trained model
y_pred = nb_classifier.predict(X_test)
y_true = y_test
# Ignore all warnings
warnings.filterwarnings("ignore")

# Train your model and perform classification
# ...

# Calculate precision and F-score with zero_division parameter
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)



# Evaluate the Naïve Bayes model
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naïve Bayes Accuracy:", nb_accuracy)
print("Classification Report for Naïve Bayes:")
print(classification_report(y_test, nb_predictions))

# Compare and select the best model based on accuracy or other evaluation metrics
if dt_accuracy > nb_accuracy:
    best_model = dt_classifier
    print("Decision Tree selected as the best model.")
else:
    best_model = nb_classifier
    print("Naïve Bayes selected as the best model.")
# -



# - And for all numeric

# +
# Prepare the data
X_all = df_new_all.drop(columns=['Attrition'])  # Features
y_all = df_new_all['Attrition']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Train the Naïve Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the Decision Tree model
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)
print("Classification Report for Decision Tree:")
print(classification_report(y_test, dt_predictions))

# Evaluate the Naïve Bayes model
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naïve Bayes Accuracy:", nb_accuracy)
print("Classification Report for Naïve Bayes:")
print(classification_report(y_test, nb_predictions))

# Compare and select the best model based on accuracy or other evaluation metrics
if dt_accuracy > nb_accuracy:
    best_model = dt_classifier
    print("Decision Tree selected as the best model.")
else:
    best_model = nb_classifier
    print("Naïve Bayes selected as the best model.")
# -

# # Unsupervised machine learning: clustering
# ## apply at least one clustering algorithm (e.g. K-Means) for segmentation of the employees in groups of similarity

# Determine k by minimizing the distortion - 
# the sum of the squared distances between each observation vector and its centroid
distortions = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10).fit(df_new)
    model.fit(df_new)
    distortions.append(sum(np.min(cdist(df_new, model.cluster_centers_, 'euclidean'), axis=1)) / df_new.shape[0]) 
print("Distortion: ", distortions)

# Plot the distortion to discover the elbow
plt.title('Elbow Method for Optimal K')
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.show()
st.pyplot(plt)

# Determine k by minimizing the distortion - 
# the sum of the squared distances between each observation vector and its centroid
distortions_all = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10).fit(df_new_all)
    model.fit(df_new_all)
    distortions_all.append(sum(np.min(cdist(df_new_all, model.cluster_centers_, 'euclidean'), axis=1)) / df_new_all.shape[0]) 
print("Distortion for all numeric: ", distortions_all)

# Plot the distortion to discover the elbow
plt.title('Elbow Method for Optimal K')
plt.plot(K, distortions_all, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.show()
st.pyplot(plt)

# Optimal number of clusters K
num_clusters = 7

# Create an instance of KMeans classifier
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)
# init: method of experimemtal finding the initial location of the centroids
# n_init: the algorithm will run n_init times with different cetroids and the best result of those will be taken

kmeans.fit(df_new)


# - Here we can really se that the silhouette score computes to 2 clusters as the best. This is of cause because the attrition only have two options 'Yes' or 1 or 'No' or 0.

# Determine k by maximising the silhouette score for each number of clusters
scores = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    print("\nNumber of clusters =", k)
    print("Silhouette score =", score)
    scores.append(score)

# - Now for all

# Determine k by maximising the silhouette score for each number of clusters
scores = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X_all)
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    print("\nNumber of clusters =", k)
    print("Silhouette score =", score)
    scores.append(score)

# - Here we can really se that the silhouette score computes to 2 clusters as the best. This is of cause because the attrition only have two options 'Yes' or 1 or 'No' or 0.

# +
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame with relevant employee data

# Step 2: Feature Selection
columns_to_keep

# Step 3: Normalization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_new[columns_to_keep])

# Step 4: Choosing the Number of Clusters (K)
# Use the Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# Based on the Elbow Method, choose the optimal number of clusters (K)

# Step 5: Model Training
k = 2  # Example: choose the number of clusters based on the elbow method
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(df_scaled)

# Step 6: Cluster Assignment
df_new['Cluster'] = kmeans.labels_

# Step 7: Cluster Analysis
cluster_summary = df_new.groupby('Cluster')[columns_to_keep].mean()
print(cluster_summary)

# Step 8: Interpretation
# Analyze the characteristics of each cluster to derive insights and recommendations
# -

df_new

# ## Here we try with all numeric columns non excepted

# +
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame with relevant employee data

# With all numeric columns

# Step 3: Normalization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_new_all[columns_to_keep_all])

# Step 4: Choosing the Number of Clusters (K)
# Use the Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# Based on the Elbow Method, choose the optimal number of clusters (K)

# Step 5: Model Training
k = 2  # Example: choose the number of clusters based on the elbow method
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(df_scaled)

# Step 6: Cluster Assignment
df_new_all['Cluster'] = kmeans.labels_

# Step 7: Cluster Analysis
cluster_summary = df_new_all.groupby('Cluster')[columns_to_keep_all].mean()
print(cluster_summary)

# Step 8: Interpretation
# Analyze the characteristics of each cluster to derive insights and recommendations
# -

df_new_all

# +
# #!pip install jupytext - using it to convert this to a .py file
# #!jupytext --output streamlitMP4.py
# -


