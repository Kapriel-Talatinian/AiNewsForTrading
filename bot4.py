import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from binance.client import Client
from binance.enums import HistoricalKlinesType
import argparse

# Argument parser for CLI parameters
parser = argparse.ArgumentParser()
parser.add_argument("--risk", type=float, default=0.1, help="Risk per trade (e.g., 0.1 for 10%)")
parser.add_argument("--leverage", type=int, default=30, help="Leverage multiplier (e.g., 10)")
args = parser.parse_args()

# Load real historical data from Binance
client = Client()
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR
start = "1 Jan, 2022"
end = "1 Jan, 2024"

klines = client.get_historical_klines(symbol, interval, start, end, klines_type=HistoricalKlinesType.SPOT)

df = pd.DataFrame(klines, columns=[
    "Time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base volume", "Taker buy quote volume", "Ignore"
])
df["Time"] = pd.to_datetime(df["Time"], unit='ms')
df.set_index("Time", inplace=True)
df = df.astype(float)[["Open", "High", "Low", "Close"]]

# Display actual dataset duration
print(f"Dataset duration: {df.index[0]} to {df.index[-1]} — Total {df.index[-1] - df.index[0]}")

# Moving averages
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Trading parameters
capital = 100000
risk_per_trade = args.risk
leverage = args.leverage
pitchfork_window = 200
variation_min_pct = 0.0005
congestion_threshold = 0.002
trailing_stop_pct = 0.25

trades = []
balance = capital
balance_history = []
profits = []

for i in range(pitchfork_window, len(df)-2):
    window = df.iloc[i-pitchfork_window:i]

    p1 = window.iloc[0]
    p2 = window.iloc[pitchfork_window//2]
    p3 = window.iloc[-1]

    slope, intercept, _, _, _ = linregress([p1.name.value, p3.name.value], [p1['Close'], p3['Close']])
    midline = slope * df.index[i].value + intercept
    diff = abs(p2['Close'] - p1['Close'])
    upper = midline + diff
    lower = midline - diff

    prev_close = df['Close'].iloc[i-1]
    current_close = df['Close'].iloc[i]
    price_variation = abs(current_close - prev_close)
    ma_diff_pct = abs(df['MA20'].iloc[i] - df['MA50'].iloc[i]) / df['Close'].iloc[i]
    rsi = df['RSI'].iloc[i]

    variation_min = df['Close'].iloc[i] * variation_min_pct
    if price_variation < variation_min or ma_diff_pct < congestion_threshold or pd.isna(rsi):
        continue

    trend_up = df['MA20'].iloc[i] > df['MA50'].iloc[i]
    trend_down = df['MA20'].iloc[i] < df['MA50'].iloc[i]

    position_size = balance * risk_per_trade * leverage

    # Breakout Up
    if prev_close <= upper and current_close > upper:
        if not trend_up or rsi < 55:
            continue

        entry_price = current_close
        sl = entry_price - (diff * 0.5)
        tp = entry_price + diff

        next_high = df['High'].iloc[i+1]
        next_low = df['Low'].iloc[i+1]

        if next_high >= tp:
            profit = position_size * ((tp - entry_price) / entry_price)
        elif next_low <= sl:
            profit = -position_size * ((entry_price - sl) / entry_price)
        else:
            continue

        balance += profit
        trades.append((df.index[i], "buy", entry_price, balance))
        balance_history.append(balance)
        profits.append(profit)

    # Breakout Down
    elif prev_close >= lower and current_close < lower:
        if not trend_down or rsi > 45:
            continue

        entry_price = current_close
        sl = entry_price + (diff * 0.5)
        tp = entry_price - diff

        next_high = df['High'].iloc[i+1]
        next_low = df['Low'].iloc[i+1]

        if next_low <= tp:
            profit = position_size * ((entry_price - tp) / entry_price)
        elif next_high >= sl:
            profit = -position_size * ((sl - entry_price) / entry_price)
        else:
            continue

        balance += profit
        trades.append((df.index[i], "sell", entry_price, balance))
        balance_history.append(balance)
        profits.append(profit)

# Drawdown calculation
balance_series = pd.Series(balance_history)
rolling_max = balance_series.cummax()
drawdowns = rolling_max - balance_series
max_drawdown = drawdowns.max()
max_drawdown_pct = (max_drawdown / capital) * 100

dynamic_drawdown = drawdowns
static_drawdown = np.maximum(0, capital - balance_series)
max_static_drawdown = static_drawdown.max()
max_static_drawdown_pct = (max_static_drawdown / capital) * 100

avg_dynamic_drawdown = drawdowns.mean()
avg_static_drawdown = static_drawdown.mean()

# Performance stats
if profits:
    win_trades = [p for p in profits if p > 0]
    lose_trades = [p for p in profits if p < 0]
    win_rate = len(win_trades) / len(profits) * 100
    profit_factor = abs(sum(win_trades) / sum(lose_trades)) if lose_trades else np.inf
    expectancy = np.mean(profits)
else:
    win_rate = profit_factor = expectancy = 0

# Plotting Equity Curve
plt.figure(figsize=(14,8))
plt.plot(balance_series.index, balance_series.values, label="Equity Curve", color='blue')
plt.plot(balance_series.index, rolling_max, label="Rolling Max", linestyle='--', color='green')
plt.plot(balance_series.index, balance_series.values - dynamic_drawdown, label="Dynamic Drawdown", linestyle=':', color='red')
plt.plot(balance_series.index, capital - static_drawdown, label="Static Drawdown", linestyle=':', color='orange')
plt.title("Equity Curve with Drawdowns")
plt.xlabel("Trades")
plt.ylabel("Balance (USD)")
plt.legend()
plt.grid()
plt.show()

# Results
print(f"Initial capital: {capital}")
print(f"Final balance: {balance:.2f}")
print(f"Total trades: {len(trades)}")
print(f"Max Dynamic Drawdown: {max_drawdown:.2f} USD ({max_drawdown_pct:.2f}%)")
print(f"Max Static Drawdown: {max_static_drawdown:.2f} USD ({max_static_drawdown_pct:.2f}%)")
print(f"Avg Dynamic Drawdown: {avg_dynamic_drawdown:.2f} USD")
print(f"Avg Static Drawdown: {avg_static_drawdown:.2f} USD")
print(f"Win rate: {win_rate:.2f}%")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Expectancy (avg P/L per trade): {expectancy:.2f} USD")
