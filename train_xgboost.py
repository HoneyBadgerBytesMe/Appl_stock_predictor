import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
data_aapl = pd.read_csv("Download Data  - STOCK_US_XNAS_AAPL.csv")
historical_quotes = pd.read_csv("HistoricalQuotes.csv")

# Clean and preprocess datasets
historical_quotes.columns = historical_quotes.columns.str.strip()
data_aapl['Date'] = pd.to_datetime(data_aapl['Date'], format='%m/%d/%Y')
data_aapl['Volume'] = data_aapl['Volume'].str.replace(',', '').astype(int)
data_aapl[['Open', 'High', 'Low', 'Close']] = data_aapl[['Open', 'High', 'Low', 'Close']].astype(float)

historical_quotes['Date'] = pd.to_datetime(historical_quotes['Date'], format='%m/%d/%Y')
for col in ['Close/Last', 'Open', 'High', 'Low']:
    historical_quotes[col] = historical_quotes[col].str.replace('[$,]', '', regex=True).astype(float)
historical_quotes.rename(columns={'Close/Last': 'Close'}, inplace=True)

merged_data = pd.concat([data_aapl, historical_quotes])
merged_data = merged_data.sort_values(by='Date').reset_index(drop=True)

# Feature engineering
merged_data['Prev_Close'] = merged_data['Close'].shift(1)
merged_data['Daily_Range'] = merged_data['High'] - merged_data['Low']
merged_data['Pct_Change'] = merged_data['Close'].pct_change() * 100
merged_data = merged_data.dropna().reset_index(drop=True)

# Define features and target
X = merged_data[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Daily_Range', 'Pct_Change']]
y = merged_data['Close']

# Sample smaller data for testing
X_small = X.sample(frac=0.2, random_state=42)
y_small = y.loc[X_small.index]

# Train and test XGBoost models with different configurations
parameter_combinations = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
]

results = []
for params in parameter_combinations:
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='rmse'
    )
    model.fit(X_small, y_small)
    y_pred = model.predict(X_small)
    mse = mean_squared_error(y_small, y_pred)
    r2 = r2_score(y_small, y_pred)
    results.append({
        'Parameters': params,
        'MSE': mse,
        'R2': r2
    })

# Display results
results_df = pd.DataFrame(results)
print("XGBoost Model Results:")
print(results_df)



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Define more parameter combinations for testing
extended_parameters = [
    {'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 1.0},
    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.8},
]

# Test each parameter combination
tuning_results = []
for params in extended_parameters:
    # Initialize and train the model with the current parameters
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='rmse'
    )
    model.fit(X_small, y_small)
    
    # Predict on the small dataset
    y_pred = model.predict(X_small)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_small, y_pred)
    r2 = r2_score(y_small, y_pred)
    
    # Store results
    tuning_results.append({
        'Parameters': params,
        'MSE': mse,
        'R2': r2
    })

# Convert results to a DataFrame for comparison
tuning_results_df = pd.DataFrame(tuning_results)

# Display tuning results
print("Extended XGBoost Model Tuning Results:")
print(tuning_results_df)
