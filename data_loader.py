import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import os
from scipy.stats import norm
import numpy as np
import seaborn as sns

# Read the CSV file into a DataFrame
def read_csv(file_path):
    return pd.read_csv(file_path)

# Clean the DataFrame
def clean_data(data):
    # Drop rows with missing values
    data_cleaned = data.dropna()
    
    # Convert the 'Date' column to a datetime object
    data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%d-%m-%Y')

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


def clean_out_features(data, features):
    # Check if the features exist in the DataFrame before dropping them
    existing_features = [col for col in features if col in data.columns]
    
    # Drop only the existing features
    data_cleaned = data.drop(existing_features, axis=1)
    return data_cleaned

    
def create_and_save_box_plot(data, column='Weekly_Sales'):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()

def create_box_plot_no_outliers(data, column='Weekly_Sales', save_path='boxplot.png', show_outliers=False):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column], showfliers=show_outliers)
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.savefig(save_path)
    plt.close()

from scipy.stats import kstest

from scipy.stats import shapiro

def normal_test(data):
    if data.empty:
        return ["DataFrame is empty. Unable to perform normality test."]
    
    # Drop NaN values from the DataFrame
    data_cleaned = data.dropna()
    
    # Convert columns to numeric type
    numeric_data = data_cleaned.apply(pd.to_numeric, errors='coerce')
    numeric_columns = numeric_data.columns
    
    if numeric_columns.empty:
        return ["No numeric columns found in the DataFrame. Unable to perform normality test."]
    
    # Check for normal distribution using the Shapiro-Wilk test
    alpha = 0.05
    results = []
    for column in numeric_columns:
        stat, p = shapiro(numeric_data[column])
        if p > alpha:
            results.append(f"Data in column '{column}' is normally distributed (p = {p:.4f})")
        else:
            results.append(f"Data in column '{column}' is not normally distributed (p = {p:.4f})")
    return results

def visualize_sales_histogram(data, save_path='sales_histogram.png'):
    # Convert 'Date' column to datetime type
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Group data by 3-month intervals and sum weekly sales
    grouped_data = data.resample('3M', on='Date')['Weekly_Sales'].sum()
    
    # Plot the histogram
    plt.bar(grouped_data.index.astype(str), grouped_data.values)
    
    plt.title('Distribution of Weekly Sales Over Time (3-Month Intervals)')
    plt.xlabel('Time Interval')
    plt.ylabel('Total Weekly Sales')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def save_correlation_heatmap(data, save_path='correlation_heatmap.png'):
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    
    # Save heatmap to a file
    plt.savefig(save_path)
    plt.close()
    
    print("Correlation heatmap saved successfully.")

import os

def visualize_sales_by_stores(data, save_path='static/sales_by_stores.png'):
    #What are the highest performing stores? (Weekly sales by stores)
    plt.figure(figsize=(15,5))
    fig = sns.barplot(x='Store', y='Weekly_Sales',  color='#FFC220', data=data)

    plt.xlabel('Store Number')
    plt.ylabel('Weekly Sales')
    plt.title('Weekly Sales by Stores')

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)

