import pandas as pd
import numpy as np
from scipy.stats import linregress
from binance.client import Client
from binance.enums import HistoricalKlinesType
import argparse

class BacktestBot:
    def __init__(self, risk, leverage, max_pyramids):
        # Paramètres de trading
        self.risk = risk
        self.leverage = leverage
        self.max_pyramids = max_pyramids
        self.capital = 100_000
        self.balance = self.capital

        # Chargement des données
        client = Client()
        symbol = "BTCUSDT"
        interval = Client.KLINE_INTERVAL_1HOUR
        start = "1 Jan, 2025"
        end = "1 May, 2025"
        klines = client.get_historical_klines(
            symbol, interval, start, end,
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

        # Indicateurs
        self.df["MA20"] = self.df["Close"].rolling(20).mean()
        self.df["MA50"] = self.df["Close"].rolling(50).mean()
        delta = self.df["Close"].diff()
        gain = delta.where(delta>0, 0).rolling(14).mean()
        loss = -delta.where(delta<0, 0).rolling(14).mean()
        self.df["RSI"] = 100 - (100 / (1 + gain/loss))

        # Pour la courbe d'equity
        pitchfork_window = 200
        first_ts = self.df.index[pitchfork_window]
        self.balance_history = [self.balance]
        self.balance_times = [first_ts]

        # État de la position
        self.position = None      # "long" / "short" / None
        self.entries = []         # liste des legs ouverts
        self.pyramid_count = 0    # nombre de pyramides effectuées
        self.stop_loss = 0.0
        self.take_profit = 0.0

        # Journal des trades fermés
        self.trades = []

    def enter(self, side, price, sl, tp, timestamp):
        """Ouvre une nouvelle tranche (position initiale ou pyramiding)."""
        leg_size = self.balance * self.risk * self.leverage
        if self.position is None:
            self.position = side
            self.pyramid_count = 0
            self.entries = []
        self.entries.append({
            "time": timestamp,
            "price": price,
            "size": leg_size
        })
        self.pyramid_count += 1
        self.stop_loss = sl
        self.take_profit = tp

    def exit_all(self, exit_price, exit_time):
        """Clôture tous les legs, calcule PnL et reset l'état."""
        for leg in self.entries:
            if self.position == "long":
                pnl = leg["size"] * ((exit_price - leg["price"]) / leg["price"])
            else:  # short
                pnl = leg["size"] * ((leg["price"] - exit_price) / leg["price"])
            self.trades.append({
                "entry_time": leg["time"],
                "exit_time": exit_time,
                "side": self.position,
                "entry_price": leg["price"],
                "exit_price": exit_price,
                "size": leg["size"],
                "pnl": pnl
            })
            self.balance += pnl

        # Mettre à jour la courbe d'equity
        self.balance_history.append(self.balance)
        self.balance_times.append(exit_time)

        # Reset de l'état de position
        self.position = None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def run(self):
        """Boucle principale de backtest."""
        pitchfork_window = 200
        variation_min_pct = 0.0005
        congestion_threshold = 0.002

        for i in range(pitchfork_window, len(self.df) - 2):
            df = self.df
            window = df.iloc[i-pitchfork_window:i]
            p1, p2, p3 = window.iloc[0], window.iloc[pitchfork_window//2], window.iloc[-1]
            slope, intercept, *_ = linregress(
                [p1.name.value, p3.name.value],
                [p1["Close"], p3["Close"]]
            )
            t = df.index[i].value
            midline = slope * t + intercept
            diff = abs(p2["Close"] - p1["Close"])
            upper, lower = midline + diff, midline - diff

            prev_close = df["Close"].iloc[i-1]
            curr_close = df["Close"].iloc[i]
            price_variation = abs(curr_close - prev_close)
            ma_diff_pct = abs(df["MA20"].iloc[i] - df["MA50"].iloc[i]) / curr_close
            rsi = df["RSI"].iloc[i]
            if price_variation < curr_close * variation_min_pct or ma_diff_pct < congestion_threshold or pd.isna(rsi):
                continue

            trend_up = df["MA20"].iloc[i] > df["MA50"].iloc[i]
            trend_down = df["MA20"].iloc[i] < df["MA50"].iloc[i]

            # Check sorties sur la bougie suivante
            next_high = df["High"].iloc[i+1]
            next_low = df["Low"].iloc[i+1]
            exit_time = df.index[i+1]
            if self.position == "long":
                if next_high >= self.take_profit:
                    self.exit_all(self.take_profit, exit_time)
                elif next_low <= self.stop_loss:
                    self.exit_all(self.stop_loss, exit_time)
            elif self.position == "short":
                if next_low <= self.take_profit:
                    self.exit_all(self.take_profit, exit_time)
                elif next_high >= self.stop_loss:
                    self.exit_all(self.stop_loss, exit_time)

            # Signal d'entrée haussier
            if prev_close <= upper < curr_close and trend_up and rsi >= 55:
                if self.position == "short":
                    continue
                sl = curr_close - diff * 0.5
                tp = curr_close + diff
                if self.position is None or self.pyramid_count < self.max_pyramids:
                    self.enter("long", curr_close, sl, tp, df.index[i])

            # Signal d'entrée baissier
            elif prev_close >= lower > curr_close and trend_down and rsi <= 45:
                if self.position == "long":
                    continue
                sl = curr_close + diff * 0.5
                tp = curr_close - diff
                if self.position is None or self.pyramid_count < self.max_pyramids:
                    self.enter("short", curr_close, sl, tp, df.index[i])

    def report(self):
        """Affichage des résultats et des statistiques sans générer de graphiques."""
        df_trades = pd.DataFrame(self.trades)
        # Sauvegarde CSV
        df_trades.to_csv("trades_log.csv", index=False)
        print("Journal des trades exporté vers trades_log.csv")

        # Statistiques générales
        total_trades = len(df_trades)
        total_pnl = df_trades["pnl"].sum() if total_trades else 0
        win_rate = df_trades["pnl"].gt(0).mean() * 100 if total_trades else 0
        avg_pnl = df_trades["pnl"].mean() if total_trades else 0
        max_win = df_trades["pnl"].max() if total_trades else 0
        max_loss = df_trades["pnl"].min() if total_trades else 0

        # Durée moyenne du trade
        if total_trades:
            durations = (df_trades["exit_time"] - df_trades["entry_time"])  
            avg_duration = durations.mean()
        else:
            avg_duration = pd.Timedelta(0)

        # Calcul drawdown et solde min
        balances = np.array(self.balance_history)
        rolling_max = np.maximum.accumulate(balances)
        drawdowns = rolling_max - balances
        max_drawdown = drawdowns.max()
        min_balance = balances.min()

        # Affichage
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
        print(f"Max drawdown        : {max_drawdown:,.2f} USD ({(max_drawdown/self.capital)*100:.2f}%)")
        print(f"Solde minimum       : {min_balance:,.2f} USD")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk", type=float, default=0.1)
    parser.add_argument("--leverage", type=int, default=30)
    parser.add_argument("--max_pyramids", type=int, default=3)
    args = parser.parse_args()

    bot = BacktestBot(args.risk, args.leverage, args.max_pyramids)
    bot.run()
    bot.report()
