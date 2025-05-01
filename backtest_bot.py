import pandas as pd
import numpy as np
from scipy.stats import linregress
from binance.client import Client
from binance.enums import HistoricalKlinesType
import argparse
import itertools

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
        start = "1 Jan, 2022"
        end = "1 Jan, 2024"
        klines = client.get_historical_klines(
            symbol, interval, start, end,
            klines_type=HistoricalKlinesType.SPOT
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

        # Courbe d'equity tracking
        pitchfork_window = 200
        first_ts = self.df.index[pitchfork_window]
        self.balance_history = [self.balance]
        self.balance_times = [first_ts]

        # État de position
        self.position = None  # "long" / "short" / None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        # Journal trades clos
        self.trades = []

    def enter(self, side, price, sl, tp, timestamp):
        """Ouvre une nouvelle tranche (position initiale ou pyramiding)."""
        leg_size = self.balance * self.risk * self.leverage
        if self.position is None:
            self.position = side
            self.pyramid_count = 0
            self.entries = []
        self.entries.append({'time': timestamp, 'price': price, 'size': leg_size})
        self.pyramid_count += 1
        self.stop_loss = sl
        self.take_profit = tp

    def exit_all(self, exit_price, exit_time):
        """Clôture tous les legs, calcule PnL, reset état."""
        for leg in self.entries:
            if self.position == "long":
                pnl = leg['size'] * ((exit_price - leg['price']) / leg['price'])
            else:
                pnl = leg['size'] * ((leg['price'] - exit_price) / leg['price'])
            self.trades.append({
                'entry_time': leg['time'],
                'exit_time': exit_time,
                'side': self.position,
                'entry_price': leg['price'],
                'exit_price': exit_price,
                'size': leg['size'],
                'pnl': pnl
            })
            self.balance += pnl

        # MAJ equity
        self.balance_history.append(self.balance)
        self.balance_times.append(exit_time)

        # reset state
        self.position = None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def run(self):
        pitchfork_window = 200
        variation_min_pct = 0.0005
        congestion_threshold = 0.002

        for i in range(pitchfork_window, len(self.df) - 2):
            df = self.df
            window = df.iloc[i-pitchfork_window:i]
            p1, p2, p3 = window.iloc[0], window.iloc[pitchfork_window//2], window.iloc[-1]
            slope, intercept, *_ = linregress(
                [p1.name.value, p3.name.value],
                [p1['Close'], p3['Close']]
            )
            t = df.index[i].value
            midline = slope * t + intercept
            diff = abs(p2['Close'] - p1['Close'])
            upper, lower = midline + diff, midline - diff

            prev_c = df['Close'].iloc[i-1]
            curr_c = df['Close'].iloc[i]
            var = abs(curr_c - prev_c)
            ma_diff = abs(df['MA20'].iloc[i] - df['MA50'].iloc[i]) / curr_c
            rsi = df['RSI'].iloc[i]
            if var < curr_c * variation_min_pct or ma_diff < congestion_threshold or pd.isna(rsi):
                continue

            up, down = df['MA20'].iloc[i] > df['MA50'].iloc[i], df['MA20'].iloc[i] < df['MA50'].iloc[i]
            nh, nl = df['High'].iloc[i+1], df['Low'].iloc[i+1]
            t_exit = df.index[i+1]

            # sortie
            if self.position == 'long':
                if nh >= self.take_profit: self.exit_all(self.take_profit, t_exit)
                elif nl <= self.stop_loss: self.exit_all(self.stop_loss, t_exit)
            elif self.position == 'short':
                if nl <= self.take_profit: self.exit_all(self.take_profit, t_exit)
                elif nh >= self.stop_loss: self.exit_all(self.stop_loss, t_exit)

            # entrée
            if prev_c <= upper < curr_c and up and rsi >= 55:
                if self.position == 'short': continue
                sl, tp = curr_c - diff*0.5, curr_c + diff
                if self.position is None or self.pyramid_count < self.max_pyramids:
                    self.enter('long', curr_c, sl, tp, df.index[i])
            elif prev_c >= lower > curr_c and down and rsi <= 45:
                if self.position == 'long': continue
                sl, tp = curr_c + diff*0.5, curr_c - diff
                if self.position is None or self.pyramid_count < self.max_pyramids:
                    self.enter('short', curr_c, sl, tp, df.index[i])

    def get_metrics(self):
        """Retourne metrics: final_balance, max_drawdown, trade count."""
        balances = np.array(self.balance_history)
        roll_max = np.maximum.accumulate(balances)
        dd = roll_max - balances
        return self.balance, dd.max(), len(self.trades)

    def report(self):
        df_trades = pd.DataFrame(self.trades)
        df_trades.to_csv('trades_log.csv', index=False)
        print('Journal des trades exporté vers trades_log.csv')
        total = len(df_trades)
        pnl = df_trades['pnl'].sum() if total else 0
        wr = df_trades['pnl'].gt(0).mean()*100 if total else 0
        avg = df_trades['pnl'].mean() if total else 0
        mxw = df_trades['pnl'].max() if total else 0
        mxl = df_trades['pnl'].min() if total else 0
        dur = (df_trades['exit_time'] - df_trades['entry_time']).mean() if total else pd.Timedelta(0)
        bal, mdd, _ = self.get_metrics()
        print('=== RESULTATS BACKTEST ===')
        print(f'Capital init : {self.capital:,.2f} USD')
        print(f'Capital final: {self.balance:,.2f} USD')
        print(f'Trades       : {total}')
        print(f'Net PnL      : {pnl:,.2f} USD')
        print(f'Win rate     : {wr:.2f}%')
        print(f'Avg PnL      : {avg:.2f} USD')
        print(f'Big win      : {mxw:.2f} USD')
        print(f'Big loss     : {mxl:.2f} USD')
        print(f'Avg duration : {dur}')
        print(f'Max DD       : {mdd:,.2f} USD ({mdd/self.capital*100:.2f}%)')


def optimize(risk_min, risk_max, lev_min, lev_max, pyr_min, pyr_max, dd_limit):
    best = {'balance': -np.inf, 'risk': None, 'leverage': None, 'pyramids': None, 'drawdown': None}
    for risk in np.linspace(risk_min, risk_max, int((risk_max-risk_min)/0.01)+1):
        for lev in range(lev_min, lev_max+1):
            for pyr in range(pyr_min, pyr_max+1):
                bot = BacktestBot(risk, lev, pyr)
                bot.run()
                bal, mdd, cnt = bot.get_metrics()
                if mdd <= dd_limit * bot.capital:
                    if bal > best['balance']:
                        best.update({'balance': bal, 'risk': risk, 'leverage': lev,
                                     'pyramids': pyr, 'drawdown': mdd})
    print('=== OPTIMISATION RESULT ===')
    print(f"Risk        : {best['risk']:.2f}")
    print(f"Leverage    : {best['leverage']}")
    print(f"Pyramids    : {best['pyramids']}")
    print(f"Final Bal   : {best['balance']:.2f} USD")
    print(f"Max Drawdown: {best['drawdown']:.2f} USD ({best['drawdown']/100000*100:.2f}% initial)")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--risk_min', type=float, default=0.1)
    p.add_argument('--risk_max', type=float, default=0.2)
    p.add_argument('--lev_min', type=int, default=10)
    p.add_argument('--lev_max', type=int, default=50)
    p.add_argument('--pyr_min', type=int, default=0)
    p.add_argument('--pyr_max', type=int, default=5)
    p.add_argument('--dd_limit', type=float, default=0.09,
                   help='Seuil max drawdown en fraction')
    p.add_argument('--optimize', action='store_true', help='Lancer optimisation')
    args = p.parse_args()

    if args.optimize:
        optimize(args.risk_min, args.risk_max,
                 args.lev_min, args.lev_max,
                 args.pyr_min, args.pyr_max,
                 args.dd_limit)
    else:
        bot = BacktestBot(args.risk_min, args.lev_min, args.pyr_min)
        bot.run()
        bot.report()
