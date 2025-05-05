import pandas as pd
import numpy as np
from scipy.stats import linregress
from binance.client import Client
from binance.enums import HistoricalKlinesType
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest Bot with detailed stats and equity limits")
    parser.add_argument("--risk", type=float, default=0.1, help="Risque par trade (fraction)")
    parser.add_argument("--leverage", type=int, default=20, help="Effet de levier")
    parser.add_argument("--max_pyramids", type=int, default=4, help="Nombre max de pyramides")
    parser.add_argument("--min_equity", type=float, default=50000.0, help="Équité minimale en USD pour arrêter")
    args = parser.parse_args()
    return args

class BacktestBot:
    def __init__(self, risk, leverage, max_pyramids, min_equity):
        self.risk = risk
        self.leverage = leverage
        self.max_pyramids = max_pyramids
        self.min_equity = min_equity
        self.capital = 50_000.0
        self.balance = self.capital

        client = Client()
        klines = client.get_historical_klines(
            "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE,
            "4 Apr, 2025", "4 May, 2025",
            klines_type=HistoricalKlinesType.FUTURES
        )
        df = pd.DataFrame(klines, columns=[
            "Time","Open","High","Low","Close","Volume",
            "Close time","Quote asset volume","Number of trades",
            "Taker buy base volume","Taker buy quote volume","Ignore"
        ])
        df["Time"] = pd.to_datetime(df["Time"], unit="ms")
        df.set_index("Time", inplace=True)
        self.df = df[["Open","High","Low","Close"]].astype(float)

        # Indicators
        self.df["MA20"] = self.df["Close"].rolling(20).mean()
        self.df["MA50"] = self.df["Close"].rolling(50).mean()
        delta = self.df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        self.df["RSI"] = 100 - (100 / (1 + gain/loss))

        # State
        self.position = None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trades = []
        self.balance_history = []

    def enter(self, side, price, sl, tp, timestamp):
        # stop if below minimum equity
        if self.balance < self.min_equity:
            print(f"Arrêt: équité {self.balance:.2f} USD < seuil min {self.min_equity:.2f} USD")
            return False
        size = self.balance * self.risk * self.leverage
        if self.position is None:
            self.position = side
            self.pyramid_count = 0
            self.entries = []
        if self.pyramid_count < self.max_pyramids:
            self.entries.append({"time": timestamp, "price": price, "size": size})
            self.pyramid_count += 1
            self.stop_loss = sl
            self.take_profit = tp
        return True

    def exit_all(self, price, timestamp):
        for leg in self.entries:
            pnl = leg['size'] * ((price - leg['price']) / leg['price']) if self.position == 'long' else leg['size'] * ((leg['price'] - price) / leg['price'])
            self.trades.append({
                'entry_time': leg['time'], 'exit_time': timestamp,
                'side': self.position, 'entry_price': leg['price'], 'exit_price': price,
                'size': leg['size'], 'pnl': pnl
            })
            self.balance += pnl
        # update equity curve
        self.balance_history.append(self.balance)
        # reset
        self.position = None
        self.entries = []
        self.pyramid_count = 0

    def run(self):
        pw = 200
        for i in range(pw, len(self.df)-1):
            now = self.df.index[i]
            prev_c = self.df['Close'].iat[i-1]
            curr_c = self.df['Close'].iat[i]
            # basic filters
            if abs(curr_c - prev_c) < curr_c * 0.0005:
                continue
            if abs(self.df['MA20'].iat[i] - self.df['MA50'].iat[i]) / curr_c < 0.002:
                continue
            window = self.df.iloc[i-pw:i]
            p1, p2, p3 = window.iloc[0], window.iloc[pw//2], window.iloc[-1]
            slope, intercept, *_ = linregress([p1.name.value, p3.name.value],[p1['Close'], p3['Close']])
            mid = slope * now.value + intercept
            diff = abs(p2['Close'] - p1['Close'])
            upper, lower = mid + diff, mid - diff

            # exit next bar
            nh, nl = self.df['High'].iat[i+1], self.df['Low'].iat[i+1]
            nt = self.df.index[i+1]
            if self.position == 'long' and (nh >= self.take_profit or nl <= self.stop_loss):
                price = self.take_profit if nh >= self.take_profit else self.stop_loss
                self.exit_all(price, nt)
            if self.position == 'short' and (nl <= self.take_profit or nh >= self.stop_loss):
                price = self.take_profit if nl <= self.take_profit else self.stop_loss
                self.exit_all(price, nt)

            # entry
            if prev_c <= upper < curr_c and self.df['MA20'].iat[i] > self.df['MA50'].iat[i] and self.df['RSI'].iat[i] >= 55:
                sl, tp = curr_c - diff*0.5, curr_c + diff
                self.enter('long', curr_c, sl, tp, now)
            elif prev_c >= lower > curr_c and self.df['MA20'].iat[i] < self.df['MA50'].iat[i] and self.df['RSI'].iat[i] <= 45:
                sl, tp = curr_c + diff*0.5, curr_c - diff
                self.enter('short', curr_c, sl, tp, now)

    def report(self):
        df_tr = pd.DataFrame(self.trades)
        df_tr.to_csv('trades_log.csv', index=False)
        total_trades = len(df_tr)
        total_pnl = df_tr['pnl'].sum() if total_trades else 0
        win_rate = df_tr['pnl'].gt(0).mean()*100 if total_trades else 0
        avg_pnl = df_tr['pnl'].mean() if total_trades else 0
        max_win = df_tr['pnl'].max() if total_trades else 0
        max_loss = df_tr['pnl'].min() if total_trades else 0
        if total_trades:
            durations = df_tr['exit_time'] - df_tr['entry_time']
            avg_duration = durations.mean()
        else:
            avg_duration = pd.Timedelta(0)
        balances = np.array(self.balance_history)
        rolling_max = np.maximum.accumulate(balances)
        drawdowns = rolling_max - balances
        max_dd = drawdowns.max()
        max_dd_pct = max_dd / self.capital * 100
        min_balance = balances.min()

        print("=== BACKTEST RÉSULTATS ===")
        print(f"Capital initial     : {self.capital:,.2f} USD")
        print(f"Capital final       : {self.balance:,.2f} USD")
        print(f"Total trades        : {total_trades}")
        print(f"Gain net            : {total_pnl:,.2f} USD")
        print(f"Taux de réussite    : {win_rate:.2f}%")
        print(f"PnL moyen/trade     : {avg_pnl:,.2f} USD")
        print(f"Plus gros gain      : {max_win:,.2f} USD")
        print(f"Plus grosse perte   : {max_loss:,.2f} USD")
        print(f"Durée moyenne trade : {avg_duration}")
        print(f"Max drawdown        : {max_dd:,.2f} USD ({max_dd_pct:.2f}%)")
        print(f"Solde minimum       : {min_balance:,.2f} USD")

if __name__ == '__main__':
    args = parse_args()
    bot = BacktestBot(args.risk, args.leverage, args.max_pyramids, args.min_equity)
    bot.run()
    bot.report()