---
title: StockSage AI Terminal
emoji: 📈
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# StockSage AI Terminal

StockSage AI Terminal is a Gradio web app for stock analysis, trade planning, and price forecasting. It combines market data from Yahoo Finance, technical indicators, a PyTorch deep-learning direction model, and a statistical price forecaster into one interactive dashboard.

This app is designed as a decision-support tool. It can estimate likely direction and future price ranges, but it cannot guarantee future market prices.

## Features

- **Live ticker analysis** using `yfinance`
- **AI trend signal** showing bullish or bearish expectation
- **Future price forecast** for selectable horizons:
  - 1 Week
  - 1 Month
  - 3 Months
  - 6 Months
- **Forecast price and expected move** displayed as dashboard cards
- **80% forecast range** to show uncertainty around the prediction
- **Interactive Plotly chart** with:
  - Candlesticks
  - MA20 and MA50 moving averages
  - Forecast path
  - Forecast uncertainty band
  - Take-profit and stop-loss levels
  - RSI and volume panels
- **Risk/reward calculator** based on capital, target profit, and max loss
- **Technical analysis metrics** including RSI, annualized volatility, volume state, and MA50 trend alignment
- **Macro Intel tab** showing recent news from Yahoo Finance
- **Company Profile tab** showing valuation, profitability, trading stats, and business summary

## Models Used

### 1. PyTorch Direction Classifier

The main AI signal uses a PyTorch neural network named `StockSageModel`.

Architecture:

- 1D Convolution layer
- LSTM layer
- Fully connected dense layer
- Sigmoid output

Input features:

- Close price
- Daily returns
- 20-day moving average
- 50-day moving average
- RSI
- Relative volume

The model predicts whether the stock price is more likely to be higher or lower over the short-term training target window. The app converts the model output into:

- Bullish or bearish signal
- AI confidence score
- Trade recommendation logic

Saved model files are stored in:

```txt
model_artifacts/
```

Important files:

```txt
model_artifacts/best_model.pth
model_artifacts/scaler.pkl
model_artifacts/metadata.json
```

### 2. Price Forecasting Model

The future price feature uses a ticker-local statistical model trained at runtime from recent historical data.

Model:

- `RidgeCV` regression
- `RobustScaler`
- scikit-learn `Pipeline`

The model predicts forward log returns for the selected time horizon, then converts that predicted return into a future dollar price.

It also calculates:

- Forecast price
- Expected percentage move
- Validation MAPE
- 80% forecast range based on validation residuals

This model is intentionally separate from the PyTorch classifier. The classifier predicts direction; the Ridge model estimates a future price path.

## How It Works

1. The user selects a ticker, capital amount, target profit, stop-loss tolerance, and time horizon.
2. The app downloads historical market data from Yahoo Finance.
3. Technical indicators are calculated.
4. The PyTorch model predicts bullish or bearish direction.
5. The price forecaster trains on the selected ticker history and predicts future price.
6. The dashboard displays the signal, forecast, chart, risk/reward metrics, news, and company fundamentals.

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

If the default Gradio port is busy, run:

```bash
GRADIO_SERVER_PORT=8060 python app.py
```

## Hugging Face Spaces

This project can be deployed as a Hugging Face Space using Gradio.

Required files:

```txt
app.py
requirements.txt
model_artifacts/
sage.png
```

Hugging Face will install dependencies from `requirements.txt` and run `app.py`.

## Notebook

The notebook `draft1.ipynb` contains the training and experimentation workflow:

- Data download
- Feature engineering
- PyTorch direction classifier training
- Model artifact saving
- Price forecast model example
- Forecast visualization

## Disclaimer

StockSage is for educational and research purposes only. Stock prices are uncertain and affected by market conditions, news, liquidity, macroeconomic events, and investor behavior. Do not treat model output as financial advice.
