import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Data
# Assuming you have a CSV file containing GDP data and some economic indicators
# You can use a sample dataset or load your own data
# Let's assume the dataset contains columns: 'Year', 'GDP', 'InterestRate', 'Inflation', 'Unemployment'

# For demonstration, let's create a synthetic dataset
np.random.seed(42)
years = np.arange(1980, 2021)
gdp = np.random.normal(50000, 10000, len(years))
interest_rate = np.random.uniform(1, 10, len(years))
inflation = np.random.uniform(0, 5, len(years))
unemployment = np.random.uniform(3, 10, len(years))

data = pd.DataFrame({
    'Year': years,
    'GDP': gdp,
    'InterestRate': interest_rate,
    'Inflation': inflation,
    'Unemployment': unemployment
})

# Step 2: Preprocess Data
# Check for missing values
print(data.isnull().sum())

# If there are missing values, fill or drop them
# For simplicity, we assume there are no missing values in this synthetic dataset

# Step 3: Feature Selection
features = data[['InterestRate', 'Inflation', 'Unemployment']]
target = data['GDP']

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Step 5: Build Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(years[-len(y_test):], y_test, label='Actual GDP', marker='o')
plt.plot(years[-len(y_pred):], y_pred, label='Predicted GDP', marker='x')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('Actual vs Predicted GDP')
plt.legend()
plt.show()

# Step 7: Forecast Future GDP
# Let's forecast GDP for the next 5 years assuming economic indicators follow a pattern
future_years = np.arange(2021, 2026)
future_interest_rate = np.random.uniform(1, 10, len(future_years))
future_inflation = np.random.uniform(0, 5, len(future_years))
future_unemployment = np.random.uniform(3, 10, len(future_years))

future_data = pd.DataFrame({
    'InterestRate': future_interest_rate,
    'Inflation': future_inflation,
    'Unemployment': future_unemployment
})

future_gdp_predictions = model.predict(future_data)

# Plot future GDP predictions
plt.figure(figsize=(10, 6))
plt.plot(future_years, future_gdp_predictions, label='Forecasted GDP', marker='o')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('Forecasted GDP for Future Years')
plt.legend()
plt.show()