#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('CO2 Emissions_Canada.csv')


# In[19]:


# Check first few rows
print(df.head())

# Check data types and missing values
print(df.info())

# Basic statistics
print(df.describe())


# In[20]:


# Check for missing values
print(df.isnull().sum())

# Since there are no missing values in this dataset, no action needed


# In[21]:


# Check how many duplicates exist
print("Total duplicates:", df.duplicated().sum())

# If you want to view them:
df[df.duplicated()]

# Drop them
df = df.drop_duplicates()
df = df.drop(columns=["Fuel Consumption Comb (mpg)"])


# In[22]:


# Create a binary feature for hybrid vehicles
df['Is_Hybrid'] = df['Model'].str.contains('HYBRID', case=False).astype(int)
df


# In[23]:


# Distribution of CO2 Emissions
plt.figure(figsize=(10,6))
sns.histplot(df['CO2 Emissions(g/km)'], bins=30, kde=True)
plt.title('Distribution of CO2 Emissions')
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Count')
plt.show()

# Top 10 vehicle makes by count
plt.figure(figsize=(12,6))
df['Make'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Vehicle Makes by Count')
plt.xlabel('Make')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[24]:


# CO2 Emissions vs Engine Size
plt.figure(figsize=(10,6))
sns.scatterplot(x='Engine Size(L)', y='CO2 Emissions(g/km)', data=df, alpha=0.6)
plt.title('CO2 Emissions vs Engine Size')
plt.show()

# Average CO2 Emissions by Vehicle Class
plt.figure(figsize=(12,6))
df.groupby('Vehicle Class')['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.title('Average CO2 Emissions by Vehicle Class')
plt.ylabel('Average CO2 Emissions (g/km)')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# CO2 Emissions by Fuel Type and Cylinders
plt.figure(figsize=(12,6))
sns.boxplot(x='Cylinders', y='CO2 Emissions(g/km)', hue='Fuel Type', data=df)
plt.title('CO2 Emissions by Number of Cylinders and Fuel Type')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
numeric_cols = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
                'Fuel Consumption Hwy (L/100 km)', 'CO2 Emissions(g/km)']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[26]:


# Top 10 highest CO2 emitting models
top_emitters = df.sort_values('CO2 Emissions(g/km)', ascending=False).head(10)[['Make', 'Model', 'Vehicle Class', 'CO2 Emissions(g/km)']]
print("Top 10 Highest CO2 Emitting Vehicles:")
print(top_emitters)

# Most fuel efficient vehicles
most_efficient = df.sort_values('Fuel Consumption Comb (L/100 km)').head(10)[['Make', 'Model', 'Vehicle Class', 'Fuel Consumption Comb (L/100 km)']]
print("\nTop 10 Most Fuel Efficient Vehicles:")
print(most_efficient)

# Hybrid vs non-hybrid comparison
print("\nHybrid vs Non-Hybrid Comparison:")
print(df.groupby('Is_Hybrid')['CO2 Emissions(g/km)'].describe())


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- Step 1: Select Features and Target ---
features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 
            'Fuel Consumption Hwy (L/100 km)', 'Is_Hybrid']
target = 'CO2 Emissions(g/km)'

X = df[features]
y = df[target]
df_encoded = pd.get_dummies(df[['Vehicle Class', 'Fuel Type', 'Transmission']], drop_first=False)
X = pd.concat([X, df_encoded], axis=1)

# --- Step 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Feature Scaling (needed for SVM) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Base Regressors ---
base_models = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    ('svm', SVR(kernel='rbf', C=150, gamma=0.1))
]

# --- Step 5: Stacking Regressor ---
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
)

# --- Step 6: Train the Model ---
stacked_model.fit(X_train_scaled, y_train)

# --- Step 7: Predict and Evaluate ---
y_pred = stacked_model.predict(X_test_scaled)


# In[28]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")


# In[29]:


plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Predicted vs. Actual CO2 Emissions")
plt.legend()
plt.show()


# In[30]:


import joblib

joblib.dump(stacked_model, 'co2_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("✅ Model and scaler saved successfully.")


# In[ ]:




