import pandas as pd
from xgboost import XGBRegressor
from flask import Flask, request, jsonify
import joblib
import yfinance as yf
import threading
import time
import requests

# Step 1: Load and Preprocess the Data
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

# Step 2: Train and Save the Model
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
            
            # Return the prediction as JSON
            return jsonify({'predicted_price': prediction[0]})
        except Exception as e:
            return jsonify({'error': str(e)})

    return app

# Step 4: Fetch Live Data and Automate Predictions
def fetch_stock_data_and_predict():
    ticker = "AAPL"  # Stock ticker symbol
    print(f"Fetching data and predicting price for {ticker}...")

    while True:
        try:
            # Fetch the latest stock data
            data = yf.download(ticker, period="1d", interval="1m")
            latest_data = data.tail(1).iloc[0]

            # Prepare input for the model
            input_data = {
                "Open": latest_data["Open"],
                "High": latest_data["High"],
                "Low": latest_data["Low"],
                "Volume": latest_data["Volume"],
                "Prev_Close": latest_data["Close"],
                "Daily_Range": latest_data["High"] - latest_data["Low"],
                "Pct_Change": (latest_data["Close"] - latest_data["Open"]) / latest_data["Open"] * 100
            }

            # Send data to the locally running API
            response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
            print(f"Prediction: {response.json()}")

        except Exception as e:
            print(f"An error occurred: {e}")

        # Wait 5 minutes before fetching data again
        time.sleep(300)

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Train and save the model
    model_file = train_and_save_model(X, y)

    # Set up the API
    app = setup_api(model_file)

    # Run the Flask app in a separate thread
    threading.Thread(target=app.run, kwargs={'debug': True}).start()

    # Start fetching live stock data and automating predictions
    fetch_stock_data_and_predict()
