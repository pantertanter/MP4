import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from scipy.stats import anderson, norm
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, pairwise_distances_argmin_min
from scipy.spatial.distance import cdist


# ------------------------------------------------Loading the data------------------------------------------------

# Read the CSV file into a DataFrame
def read_csv(file_path):
    return pd.read_csv(file_path)

# ------------------------------------------------Cleaning the data------------------------------------------------

# Clean the DataFrame
def clean_data(data):
    # Drop rows with missing values
    data_cleaned = data.dropna()

    # Convert the 'Attrition' column to a binary numeric column
    data_cleaned['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

    # Convert the 'Date' column to a datetime object
    # data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%d-%m-%Y')

    # Remove outliers
    numeric_columns = data_cleaned.select_dtypes(include='number').columns
    
    for column in numeric_columns:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = data_cleaned[column].quantile(0.25)
        Q3 = data_cleaned[column].quantile(0.75)
        
        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers from the column
        data_cleaned = data_cleaned[(data_cleaned[column] >= lower_bound) & (data_cleaned[column] <= upper_bound)]
    
    return data_cleaned

# ------------------------------------------------Make colums numeric------------------------------------------------

def reverse_map_categorical_to_numeric(data):
    """
    Reverse maps categorical labels to their corresponding numeric values based on the provided mappings,
    leaving the rest of the columns unchanged.

    Parameters:
        data (DataFrame): Input DataFrame containing categorical columns.

    Returns:
        DataFrame: DataFrame with specified categorical columns replaced by their corresponding numeric values,
        while leaving the rest of the columns unchanged.
    """
    mappings = {
        'Education': {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5},
        'EnvironmentSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'JobInvolvement': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'JobSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'PerformanceRating': {'Low': 1, 'Good': 2, 'Excellent': 3, 'Outstanding': 4},
        'RelationshipSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'WorkLifeBalance': {'Bad': 1, 'Good': 2, 'Better': 3, 'Best': 4}
    }

    # Apply mappings to the specified columns
    for column, mapping in mappings.items():
        if column in data.columns:
            data[column] = data[column].map(mapping)

    return data

# ------------------------------------------------Clean out features------------------------------------------------

def clean_out_features(data, features):
    # Check if the features exist in the DataFrame before dropping them
    existing_features = [col for col in features if col in data.columns]
    
    # Drop only the existing features
    data_cleaned = data.drop(existing_features, axis=1)
    return data_cleaned

    
# ------------------------------------------------Normality test------------------------------------------------

# def normal_test(data):
#     if data.empty:
#         return ["DataFrame is empty. Unable to perform normality test."]
    
#     # Convert columns to numeric type
#     numeric_data = data.apply(pd.to_numeric, errors='coerce')
#     numeric_columns = numeric_data.columns
    
#     if numeric_columns.empty:
#         return ["No numeric columns found in the DataFrame. Unable to perform normality test."]
    
#     # Check for normal distribution using the Anderson-Darling test
#     alpha = 0.05
#     results = []
#     for column in numeric_columns:
#         result = anderson(numeric_data[column].dropna())
#         statistic = result.statistic
#         critical_values = result.critical_values
#         p_value = result.significance_level[0]  # Extracting the p-value
#         if all(statistic < critical_values):
#             results.append(f"Data in column '{column}' is normally distributed (p = {p_value:.4f})")
#         else:
#             results.append(f"Data in column '{column}' is not normally distributed (p = {p_value:.4f})")
#     return results

# Make this work for non numeric columns..

# ------------------------------------------------Visualizing the normal distribution interactive------------------------------------------------

def plot_normal_distribution(data, column_name):
    # Extract the column data
    column_data = data[column_name].dropna()
    
    # Fit a normal distribution to the data
    mu, std = norm.fit(column_data)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram of the data
    ax.hist(column_data, bins=30, density=True, alpha=0.6, color='g')

    # Plot the PDF (Probability Density Function) of the fitted normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    ax.set_title(f"Histogram and Normal Distribution Fit for {column_name}")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    ax.grid(True)

    # Display the figure in Streamlit
    st.pyplot(fig)

    # TODO make input numeric for non numeric columns

# ------------------------------------------------Creating an interactive box plot------------------------------------------------

def create_box_plot(data):
    # Generate a unique key for the selectbox
    selectbox_key = "select_column_box_plot"
    
    # Create a dropdown menu for column selection
    selected_column = st.selectbox("Select a column:", data.columns, key=selectbox_key)
    
    # Create the box plot based on the selected column
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data[selected_column])
    ax.set_title(f'Box Plot of {selected_column}')
    ax.set_ylabel(selected_column)
    return fig    

#  TODO There are some outliers. Maybe remove them?

# ------------------------------------------------Decision Tree vs. Naïve Bayes------------------------------------------------
    
def train_and_evaluate_models(X, y):
    """
    Trains and evaluates Decision Tree and Naïve Bayes classifiers.

    Parameters:
        X (DataFrame): Features.
        y (Series): Target variable.

    Returns:
        classifier: Best performing classifier (Decision Tree or Naïve Bayes).
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)

    # Train the Naïve Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Evaluate the Decision Tree model
    dt_predictions = dt_classifier.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    st.write("Decision Tree Accuracy:", dt_accuracy)
    st.write("Classification Report for Decision Tree:")
    st.write(classification_report(y_test, dt_predictions))

    # Evaluate the Naïve Bayes model
    nb_predictions = nb_classifier.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    st.write("Naïve Bayes Accuracy:", nb_accuracy)
    st.write("Classification Report for Naïve Bayes:")
    st.write(classification_report(y_test, nb_predictions))

    # Compare and select the best model based on accuracy or other evaluation metrics
    if dt_accuracy > nb_accuracy:
        best_model = dt_classifier
        st.write("Decision Tree selected as the best model.")
    else:
        best_model = nb_classifier
        st.write("Naïve Bayes selected as the best model.")

    return best_model


# ------------------------------------------------Creating a heatmap------------------------------------------------

def create_correlation_heatmap(data):
    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Return the heatmap plot
    return plt.gcf()

# ------------------------------------------------KMeans Clustering------------------------------------------------

def determine_optimal_k(data):
    """
    Determines the optimal number of clusters (k) using the elbow method.

    Parameters:
        data (DataFrame): Input DataFrame containing the data for clustering.

    Returns:
        int: Optimal number of clusters.
    """
    distortions = []
    K = range(2, 10)
    
    for k in K:
        model = KMeans(n_clusters=k, n_init=10).fit(data)
        distortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    # Find the optimal k by minimizing the distortion
    optimal_k = np.argmin(distortions) + 2

    # Plot the elbow plot
    plt.title('Elbow Method for Optimal K')
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Distortion')
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)  # Display the plot in Streamlit

    return optimal_k

st.write("I think you might be able to discuss how if 2 or ten is the optimal score ")

# ------------------------------------------------Main function------------------------------------------------