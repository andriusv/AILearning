import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

housing = pd.read_csv("housing.csv")

housing.head()
housing.info()
housing.describe()
housing.isnull().sum()

# Visualize
housing.hist(figsize=(12, 10), bins=20)
plt.show()

plt.scatter(housing['size'], housing['price'])
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Price vs Size")
plt.show()

# Scatter plot
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Price vs Size")
plt.show()

# Boxplot
sns.boxplot(x='neighborhood', y='price', data=housing)
plt.title("Price distribution by neighborhood")
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
corr = housing.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Data cleaning/preprocesing
housing_cleaned = housing.dropna()
housing_encoded = pd.get_dummies(housing_cleaned, columns=['neighborhood'], drop_first=True)

# Basic analysis
# Most expensive
housing.nlargest(5, 'price')
# Cheapest
housing.nsmallest(5, 'price')
# Statistics
housing['price'].mean()
housing['price'].median()
housing['price'].min()
housing['price'].max()

# 1. I found the EDA visualization part challenging because it's hard to choose the right plots. I solved it by experimenting with plots. Learned that data should be explored carefully.

# 2. Usually I strive for the highest score, but this time I think is G because of lack of time. Will try to fix n Jupyter Notebooks

# 3. Found the exercises very helpful and interesting. Thank you!