import pandas as pd
import numpy as np
from scipy.stats import linregress
import MetaTrader5 as mt5
import argparse
from datetime import datetime

class BacktestBotMT5:
    def __init__(self, symbol, timeframe, start, end, risk, leverage, max_pyramids):
        # Config du backtest
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start      # datetime
        self.end = end          # datetime
        self.risk = risk
        self.leverage = leverage
        self.max_pyramids = max_pyramids
        self.capital = 100_000
        self.balance = self.capital

        # Connexion MT5
        mt5.initialize()
        mt5.symbol_select(self.symbol, True)

        # Chargement des données
        rates = mt5.copy_rates_range(self.symbol, self.timeframe,
                                     self.start, self.end)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        self.df = df[['open','high','low','close']].rename(
            columns={'open':'Open','high':'High','low':'Low','close':'Close'}).astype(float)

        # Indicateurs
        self.df['MA20'] = self.df['Close'].rolling(20).mean()
        self.df['MA50'] = self.df['Close'].rolling(50).mean()
        delta = self.df['Close'].diff()
        gain = delta.where(delta>0, 0).rolling(14).mean()
        loss = -delta.where(delta<0, 0).rolling(14).mean()
        self.df['RSI'] = 100 - (100 / (1 + gain/loss))

        # État de position
        self.position = None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.balance_history = [self.balance]
        self.trades = []

    def enter(self, side, price, sl, tp, time):
        leg_size = self.balance * self.risk * self.leverage
        if self.position is None:
            self.position = side
            self.pyramid_count = 0
            self.entries = []
        self.entries.append({'time': time, 'price': price, 'size': leg_size})
        self.pyramid_count += 1
        self.stop_loss = sl
        self.take_profit = tp

    def exit_all(self, price, time):
        for leg in self.entries:
            if self.position == 'long':
                pnl = leg['size'] * ((price - leg['price']) / leg['price'])
            else:
                pnl = leg['size'] * ((leg['price'] - price) / leg['price'])
            self.trades.append({
                'entry_time': leg['time'],
                'exit_time': time,
                'side': self.position,
                'entry_price': leg['price'],
                'exit_price': price,
                'size': leg['size'],
                'pnl': pnl
            })
            self.balance += pnl
        self.balance_history.append(self.balance)
        # reset
        self.position = None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def run(self):
        pw = 200
        var_min = 0.0005
        cong_th = 0.002
        df = self.df
        for i in range(pw, len(df)-2):
            window = df.iloc[i-pw:i]
            p1, p2, p3 = window.iloc[0], window.iloc[pw//2], window.iloc[-1]
            slope, intercept, *_ = linregress(
                [p1.name.value, p3.name.value],
                [p1['Close'], p3['Close']]
            )
            t = df.index[i].value
            mid = slope*t + intercept
            diff = abs(p2['Close'] - p1['Close'])
            upper, lower = mid + diff, mid - diff

            prev_c = df['Close'].iloc[i-1]
            curr_c = df['Close'].iloc[i]
            if abs(curr_c - prev_c) < curr_c * var_min: continue
            if abs(df['MA20'].iloc[i] - df['MA50'].iloc[i]) / curr_c < cong_th: continue
            rsi = df['RSI'].iloc[i]
            up = df['MA20'].iloc[i] > df['MA50'].iloc[i]
            nh, nl = df['High'].iloc[i+1], df['Low'].iloc[i+1]
            t_next = df.index[i+1]
            # sorties
            if self.position == 'long':
                if nh >= self.take_profit:
                    self.exit_all(self.take_profit, t_next)
                elif nl <= self.stop_loss:
                    self.exit_all(self.stop_loss, t_next)
            elif self.position == 'short':
                if nl <= self.take_profit:
                    self.exit_all(self.take_profit, t_next)
                elif nh >= self.stop_loss:
                    self.exit_all(self.stop_loss, t_next)
            # entrées
            if prev_c <= upper < curr_c and up and rsi >= 55:
                if self.position == 'short': continue
                sl, tp = curr_c - diff*0.5, curr_c + diff
                if self.position is None or self.pyramid_count < self.max_pyramids:
                    self.enter('long', curr_c, sl, tp, df.index[i])
            elif prev_c >= lower > curr_c and not up and rsi <= 45:
                if self.position == 'long': continue
                sl, tp = curr_c + diff*0.5, curr_c - diff
                if self.position is None or self.pyramid_count < self.max_pyramids:
                    self.enter('short', curr_c, sl, tp, df.index[i])

    def report(self):
        df_tr = pd.DataFrame(self.trades)
        df_tr.to_csv('mt5_trades_log.csv', index=False)
        balances = np.array(self.balance_history)
        drawdown = (np.maximum.accumulate(balances) - balances).max()
        print('=== MT5 BACKTEST RESULTATS ===')
        print(f'Capital initial: {self.capital}')
        print(f'Capital final  : {self.balance:.2f}')
        print(f'Trades         : {len(df_tr)}')
        print(f'Max drawdown   : {drawdown:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk', type=float, default=0.1)
    parser.add_argument('--leverage', type=int, default=30)
    parser.add_argument('--max_pyramids', type=int, default=3)
    parser.add_argument('--symbol', type=str, default='BTCUSD')
    parser.add_argument('--timeframe', type=int, default=mt5.TIMEFRAME_H1)
    parser.add_argument('--start', type=str, default='2025-01-01')
    parser.add_argument('--end', type=str, default='2025-05-01')
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    bot = BacktestBotMT5(
        args.symbol, args.timeframe, start, end,
        args.risk, args.leverage, args.max_pyramids
    )
    bot.run()
    bot.report()
