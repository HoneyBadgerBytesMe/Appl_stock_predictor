import pandas as pd
from xgboost import XGBRegressor
from flask import Flask, request, jsonify
import joblib

# Load and Preprocess the Data
def load_and_preprocess_data():
    # Load datasets
    data_aapl = pd.read_csv("Download Data - STOCK_US_XNAS_AAPL.csv")
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

    # Merge datasets
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

    return X, y

# Training and Save the Model
def train_and_save_model(X, y):
    # Train the best model
    best_model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='rmse'
    )
    best_model.fit(X, y)  # Train on the full dataset

    # Save the model
    model_filename = "best_xgboost_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"Model saved as {model_filename}")
    return model_filename

# Step 3: Set Up the API
def setup_api(model_filename):
    # Load the trained model
    model = joblib.load(model_filename)

    # Initialize Flask app
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        # Get data from the POST request
        data = request.get_json()
        try:
            # Convert data into a DataFrame
            input_data = pd.DataFrame([data])
            
            # Predict using the model
            prediction = model.predict(input_data)
            
            # Convert prediction to float and return
            return jsonify({'predicted_price': float(prediction[0])})
        except Exception as e:
            return jsonify({'error': str(e)})

    return app

# Main Execution
if __name__ == '__main__':
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Train and save the model
    model_file = train_and_save_model(X, y)

    # Set up the API
    app = setup_api(model_file)

    # Run the API
    app.run(debug=True)
