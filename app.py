import gradio as gr
import pandas as pd
import pickle
import numpy as np
import yfinance as yf

with open("best_price_model.pkl", "rb") as f:
    model = pickle.load(f)

SEQ_LEN = 60  # lag lenght
def predict_next_day_price():
    # download latest daily AAPL data
    df = yf.download(
        "AAPL",
        period="1y",
        interval="1d",
        progress=False
    )

    df = df[["Close"]].dropna()
    last_price = float(df["Close"].iloc[-1])
    # prepare input features of last 60 day that is downloaded
    X_input = df["Close"].iloc[-SEQ_LEN:].values.reshape(1, -1)
    # predict next day price
    prediction = float(model.predict(X_input)[0])
    # difference calculation
    change = prediction - last_price
    pct_change = (change / last_price) * 100

    return (
        f"Apple (AAPL) Next-Day Prediction\n\n"
        f"Last Close Price: ${last_price:.2f}\n"
        f"Predicted Next-Day Close: ${prediction:.2f}\n"
        f"Expected Change: {change:+.2f} ({pct_change:+.2f}%)"
    )

inputs = []  

app = gr.Interface(
    fn=predict_next_day_price,
    inputs=inputs,
    outputs=gr.Textbox(lines=8, label="Prediction Result"),
    title="AAPL Next-Day Stock Price Predictor",
    description="Click Predict to fetch the latest Apple stock price and predict the next trading day's closing price."
)


app.launch()