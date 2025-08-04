# preprocessing.py
# Task 1 - Titanic Dataset: Data Cleaning & Preprocessing
# Created by Priya Vishwakarma

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv("TitanicDataset.csv")  # Make sure this file is in the same folder
print("📥 Initial Data Loaded:\n", df.head())

# Step 2: Basic Exploration
print("\n🔍 Dataset Info:")
print(df.info())
print("\n🧼 Missing Values:\n", df.isnull().sum())

# Step 3: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Too many missing values

# Step 4: Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 5: Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 6: Visualize and remove outliers in Fare
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers (Before Removal)")
plt.savefig("fare_outliers.png")  # Save the boxplot as an image
plt.close()

# Remove top 5% Fare outliers
df = df[df['Fare'] < df['Fare'].quantile(0.95)]

# Step 7: Save the cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_titanic.csv'")
print("📷 Boxplot saved as 'fare_outliers.png'")
