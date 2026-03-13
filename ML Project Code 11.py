import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load uploaded CoinGecko files
df1 = pd.read_csv("coin_gecko_2022-03-16.csv")
df2 = pd.read_csv("coin_gecko_2022-03-17.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Convert columns
numeric_cols = ["price", "1h", "24h", "7d", "24h_volume", "mkt_cap"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["coin", "date"]).copy()

# Feature engineering
df["liquidity_ratio"] = df["24h_volume"] / df["mkt_cap"]
df["volatility_proxy"] = df[["1h", "24h", "7d"]].abs().mean(axis=1)
df["price_change_day"] = df.groupby("coin")["price"].pct_change(fill_method=None)
df["volume_change_day"] = df.groupby("coin")["24h_volume"].pct_change(fill_method=None)
df["is_second_day"] = (df["date"] == df["date"].max()).astype(int)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Keep rows with valid target
df = df[df["liquidity_ratio"].notna()].copy()

# Features and target
features = [
    "price", "1h", "24h", "7d", "24h_volume", "mkt_cap",
    "volatility_proxy", "price_change_day", "volume_change_day", "is_second_day"
]
X = df[features]
y = df["liquidity_ratio"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Train final model
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42
)
model.fit(X_train_imp, y_train)

# Predict
pred = model.predict(X_test_imp)

# Evaluate
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

# Feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)
