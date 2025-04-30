import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Simulated dataset for one year (hourly)
np.random.seed(42)
dates = pd.date_range(start="2018-01-01", periods=8760, freq="H")
prices = np.cumsum(np.random.normal(0, 70, size=len(dates))) + 25000

# Create DataFrame
df = pd.DataFrame({"Close": prices}, index=dates)
df["High"] = df["Close"] + np.random.normal(30, 5, len(df))
df["Low"] = df["Close"] - np.random.normal(30, 5, len(df))
df["Open"] = df["Close"].shift(1)
df.dropna(inplace=True)

# Moving averages
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

# Trading parameters
capital = 100000
risk_per_trade = 0.1  # 4% risk per trade
leverage = 10  # default leverage
pitchfork_window = 200
variation_min_pct = 0.0005  # 0.05%

trades = []
balance = capital
balance_history = []
profits = []

for i in range(pitchfork_window, len(df)-1):
    window = df.iloc[i-pitchfork_window:i]

    # Recalculate dynamic pitchfork
    p1 = window.iloc[0]
    p2 = window.iloc[pitchfork_window//2]
    p3 = window.iloc[-1]

    slope, intercept, _, _, _ = linregress([p1.name.value, p3.name.value], [p1['Close'], p3['Close']])
    midline = slope * df.index[i].value + intercept
    diff = abs(p2['Close'] - p1['Close'])
    upper = midline + diff
    lower = midline - diff

    # Market condition
    prev_close = df['Close'].iloc[i-1]
    current_close = df['Close'].iloc[i]
    price_variation = abs(current_close - prev_close)

    variation_min = df['Close'].iloc[i] * variation_min_pct

    trend_up = df['MA20'].iloc[i] > df['MA50'].iloc[i]
    trend_down = df['MA20'].iloc[i] < df['MA50'].iloc[i]

    # Confirmed breakouts with dynamic threshold
    if price_variation < variation_min:
        continue

    position_size = balance * risk_per_trade * leverage

    # Breakout Up
    if prev_close <= upper and current_close > upper:
        if not trend_up:
            continue

        entry_price = current_close
        sl = entry_price - (diff * 0.7)
        tp = entry_price + (diff * 1.2)

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
        if not trend_down:
            continue

        entry_price = current_close
        sl = entry_price + (diff * 0.7)
        tp = entry_price - (diff * 1.2)

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
max_drawdown_pct = (max_drawdown / rolling_max.max()) * 100

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
plt.fill_between(balance_series.index, balance_series.values, rolling_max, color='red', alpha=0.3, label='Drawdown')
plt.title("Equity Curve with Drawdowns")
plt.xlabel("Trades")
plt.ylabel("Balance (USD)")
plt.legend()
plt.grid()
plt.show()

# Results
print(f"Initial capital: 100000")
print(f"Final balance: {balance:.2f}")
print(f"Total trades: {len(trades)}")
print(f"Max Drawdown: {max_drawdown:.2f} USD ({max_drawdown_pct:.2f}%)")
print(f"Win rate: {win_rate:.2f}%")
print(f"Profit factor: {profit_factor:.2f}")
print(f"Expectancy (avg P/L per trade): {expectancy:.2f} USD")
