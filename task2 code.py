# Task 2: Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset (adjust path if needed)
df = pd.read_csv("E:\world_bank_data_2025.csv")

# Step 2: Generate Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

print("\nMedian Values:")
print(df.median(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

# Step 3: Visualizing Distributions (Histograms and Boxplots)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Histograms for numeric features
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Histogram & Density of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Boxplots for numeric features
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Step 4: Feature Relationships — Correlation Matrix & Pairplot
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Features")
plt.show()

# Pairplot (use a sample if large)
sns.pairplot(df[num_cols].sample(200))
plt.show()

# Step 5: Identify Patterns, Trends, Anomalies

# Checking for duplicates
print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Inferences:
# From histograms → check for skewness
# From boxplots → identify outliers
# From heatmap → identify highly correlated variables (corr > 0.8 or < -0.8)
# From pairplot → see clusters or linear trends

# Example: finding highly correlated features
corr_matrix = df.corr(numeric_only=True)
high_corr = corr_matrix[(corr_matrix > 0.8) & (corr_matrix != 1.0)]
print("\nHighly Correlated Feature Pairs (Corr > 0.8):")
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))