import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load dataset
df = pd.read_csv('commodity_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Simulate yield (replace with real data if available)
np.random.seed(42)
df['Yield'] = df['Price'] * np.random.uniform(0.2, 0.4, size=len(df))

df = df.ffill()


# One-hot encode commodity
df = pd.get_dummies(df, columns=['Commodity'], drop_first=True)

# Feature scaling
features = ['Temperature', 'Rainfall', 'Month', 'Year'] + [col for col in df.columns if 'Commodity_' in col]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split data
X = df[features]
y_price = df['Price']
y_yield = df['Yield']

X_train, X_test, y_train_price, y_test_price, y_train_yield, y_test_yield = train_test_split(
    X, y_price, y_yield, test_size=0.2, random_state=42
)

# Train models
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model = RandomForestRegressor(n_estimators=100, random_state=42)

price_model.fit(X_train, y_train_price)
yield_model.fit(X_train, y_train_yield)

# Save models and metadata
joblib.dump(price_model, 'price_model.pkl')
joblib.dump(yield_model, 'yield_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Models trained and saved successfully!")

# Basic formula-based fallback prediction function
def predict_price(commodity, temperature, rainfall, yield_value):
    base_prices = {
        'Onion': 12,
        'Potato': 10,
        'Pulses': 50,
        'Tomato': 14,
        'Rice': 25,
        'Wheat': 22,
        'Sugar': 20
    }

    base = base_prices.get(commodity, 20)

    # Simple prediction formula
    price = base + temperature * 0.1 + rainfall * 0.2 + yield_value * 0.3

    return round(price, 2), round(yield_value, 2)
