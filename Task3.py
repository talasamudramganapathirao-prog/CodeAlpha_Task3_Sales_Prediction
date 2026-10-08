import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Sample Car Price Dataset
data = {
    'Year': [2014, 2013, 2017, 2011, 2014, 2018],
    'Present_Price': [5.59, 9.54, 9.85, 4.15, 6.87, 12.5],
    'Kms_Driven': [27000, 43000, 6900, 52000, 42450, 20000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel', 'Diesel'],
    'Selling_Price': [3.35, 4.75, 7.25, 2.85, 4.60, 8.75]
}

df = pd.DataFrame(data)

# Preprocessing & One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Features & Target Variable
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Car Price Prediction Model Results:")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
