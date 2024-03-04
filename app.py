import streamlit as st
from data_loader import (read_csv, clean_out_features,
                         plot_normal_distribution, create_box_plot,
                         create_correlation_heatmap, clean_data,
                         train_and_evaluate_models, determine_optimal_k)
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist

#------------------------------------------------Loading the data------------------------------------------------

# Read the CSV file
data = read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

#------------------------------------------------Cleaning the data------------------------------------------------

data = clean_data(data)

#------------------------------------------------Cleaning out features------------------------------------

features_to_remove = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']    

data = clean_out_features(data, features_to_remove)

# Display a sample of the data
st.write('Sample of the data:')

st.title('Attrition Data Visualization')
st.write('This is a a sample of the data:')
st.write(data.sample(n=5))

# ------------------------------------------------Make colums numeric------------------------------------------------

# # Apply reverse mapping
# data = reverse_map_categorical_to_numeric(data)

# st.write(data.isnull().sum())

#------------------------------------------------Sample plus row select-------------------------------------------

# Display a sample of the data
st.write('Sample of the data:')

st.title('Attrition Data Visualization')
st.write('This is a a sample of the data:')
st.write(data.sample(n=5))

# Check the length of the DataFrame
num_rows = len(data)
num_cols = data.shape[1]
# Write it to the screen
st.write(f'The DataFrame has {num_rows} rows and {num_cols} columns.')

# Create a number input widget to select the row number
selected_row_number = st.number_input('Enter the row number:', min_value=0, max_value=len(data), value=1, step=1)

# Check if the user has submitted the form
if st.button('Submit'):
    if selected_row_number >= 0 and selected_row_number <= len(data):
        selected_row = data.iloc[selected_row_number].to_frame().transpose()
        st.write('Selected row:')
        st.write(selected_row)
    else:
        st.write(f"Row number must be between 1 and {len(data)}.")

# ------------------------------------------------Describe------------------------------------------------

# Display the descriptive statistics
data_descriptive = data.describe()
st.write('Descriptive statistics:')
st.write(data_descriptive)

#------------------------------------------------Normality test------------------------------------------------

# # Perform the normality test
# st.write('Normality test:')
# normal_test_result = normal_test(data)

# # Display the result
# for result in normal_test_result:
#     st.write(result)

# st.write('The normality test suggests that non of the columns is normally distributed.')
# st.write('We personally do not trust this so we will make a visual representation of the normal distribution.')
# st.write('We will visualize the normal distribution of the columns to confirm the result.')

# TODO make this work

# ---------------------------Visualizing the normal distribution interactive-----------------------------------

# Optional column selection 
st.title("Normal Distribution Visualization")

# Selectbox for choosing the column
selected_column = st.selectbox("Select a column:", data.columns)

# Plot the normal distribution if a column is selected
if selected_column:
    st.write(f"### {selected_column}")
    plot_normal_distribution(data, selected_column)

st.markdown('---')
st.markdown('## Box Plot of Weekly Sales With Outliers')
st.markdown('The box plot below shows the distribution of the Weekly Sales with outliers and we can se that all the outliers are beyond the max.')
box_plot_fig = create_box_plot(data)
st.pyplot(box_plot_fig)
st.markdown('---')

# ------------------------------------------Creating a heatmap  -------------------------------------------

# st.markdown('## Correlation Heatmap With All Features')
# correlation_heatmap_fig = create_correlation_heatmap(data)
# st.pyplot(correlation_heatmap_fig)

# st.write(data.sample(n=5))

# ------------------------------------------------Decision Tree vs. Naïve Bayes------------------------------------------------

X = data.drop(columns=['Attrition'])  # Features
y = data['Attrition']  # Target variable

# Train and evaluate models
best_model = train_and_evaluate_models(X, y)

st.write(f"The best performing model is both of them as they score the same")

# ------------------------------------------------Creating an interactive box plot------------------------------------------------

optimal_k = determine_optimal_k(data)
st.write("Optimal number of clusters:", optimal_k)

# ------------------------------------------------Creating an interactive box plot------------------------------------------------

# Determine k by minimizing the distortion - 
# the sum of the squared distances between each observation vector and its centroid
distortions = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10).fit(data)
    model.fit(data)
    distortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]) 
st.write("Distortion: ", distortions)

# ------------------------------------------------Siluette score------------------------------------------------
scores = []
K = range(2, 10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    st.write("\nNumber of clusters =", k)
    st.write("Silhouette score =", score)
    scores.append(score)