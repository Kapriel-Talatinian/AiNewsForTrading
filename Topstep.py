import pandas as pd
import numpy as np
from scipy.stats import linregress
from binance.client import Client
from binance.enums import HistoricalKlinesType
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest Bot with equity and detailed stats")
    parser.add_argument("--risk", type=float, default=0.1, help="Risque par trade (fraction)")
    parser.add_argument("--leverage", type=int, default=30, help="Effet de levier")
    parser.add_argument("--max_pyramids", type=int, default=5, help="Nombre max de contrats")
    parser.add_argument("--max_daily_dd", type=float, default=1000.0, help="Drawdown journalier max en USD")
    parser.add_argument("--max_total_dd", type=float, default=2000.0, help="Drawdown total max en USD")
    parser.add_argument("--objective", type=float, default=3000.0, help="Objectif de gain en USD")
    return parser.parse_args()

class BacktestBot:
    def __init__(self, risk, leverage, max_pyramids,
                 max_daily_dd, max_total_dd, objective):
        self.risk = risk
        self.leverage = leverage
        self.max_pyramids = max_pyramids
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        self.objective = objective
        self.capital = 50_000.0
        self.balance = self.capital

        client = Client()
        klines = client.get_historical_klines(
            "BTCUSDT", Client.KLINE_INTERVAL_1HOUR,
            "1 Jan, 2018", "1 May, 2025",
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

        self.df["MA20"] = self.df["Close"].rolling(20).mean()
        self.df["MA50"] = self.df["Close"].rolling(50).mean()
        d = self.df["Close"].diff()
        g = d.where(d>0, 0).rolling(14).mean()
        l = -d.where(d<0, 0).rolling(14).mean()
        self.df["RSI"] = 100 - (100 / (1 + g/l))

        self.position = None
        self.entries = []
        self.pyramid_count = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trades = []
        self.balance_history = []
        self.equity_history = []
        self.times = []

        self.current_day = self.df.index[0].date()
        self.daily_high = self.capital
        self.all_time_high = self.capital

    def record_equity(self, time):
        unreal = 0.0
        price = self.df.loc[time, 'Close']
        for leg in self.entries:
            if self.position == 'long':
                unreal += leg['size'] * ((price - leg['price']) / leg['price'])
            else:
                unreal += leg['size'] * ((leg['price'] - price) / leg['price'])
        equity = self.balance + unreal
        self.equity_history.append(equity)
        self.balance_history.append(self.balance)
        self.times.append(time)

    def enter(self, side, price, sl, tp, timestamp):
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

    def exit_all(self, price, timestamp):
        for leg in self.entries:
            pnl = (leg['size']*((price-leg['price'])/leg['price'])) if self.position=='long' else (leg['size']*((leg['price']-price)/leg['price']))
            dur = timestamp - leg['time']
            self.trades.append({
                'entry_time': leg['time'], 'exit_time': timestamp,
                'side': self.position, 'entry_price': leg['price'], 'exit_price': price,
                'size': leg['size'], 'pnl': pnl, 'duration': dur
            })
            self.balance += pnl
        self.position=None; self.entries=[]; self.pyramid_count=0; self.stop_loss=0.0; self.take_profit=0.0

    def run(self):
        pw, var_min, cong_th = 200, 0.0005, 0.002
        for i in range(pw, len(self.df)-1):
            now = self.df.index[i]
            prev_c = self.df['Close'].iat[i-1]
            curr_c = self.df['Close'].iat[i]
            if abs(curr_c-prev_c) < curr_c*var_min: continue
            if abs(self.df['MA20'].iat[i]-self.df['MA50'].iat[i])/curr_c<cong_th: continue
            w=self.df.iloc[i-pw:i]; p1,p2,p3=w.iloc[0],w.iloc[pw//2],w.iloc[-1]
            slope,inter,*_=linregress([p1.name.value,p3.name.value],[p1['Close'],p3['Close']])
            mid=slope*now.value+inter; d=abs(p2['Close']-p1['Close'])
            upper,lower=mid+d,mid-d; nh,nl=self.df['High'].iat[i+1],self.df['Low'].iat[i+1]; nt=self.df.index[i+1]
            # exit
            if self.position=='long' and (nh>=self.take_profit or nl<=self.stop_loss):
                self.exit_all(self.take_profit if nh>=self.take_profit else self.stop_loss, nt)
            elif self.position=='short' and (nl<=self.take_profit or nh>=self.stop_loss):
                self.exit_all(self.take_profit if nl<=self.take_profit else self.stop_loss, nt)
            # entry
            if prev_c<=upper<curr_c and self.df['MA20'].iat[i]>self.df['MA50'].iat[i] and self.df['RSI'].iat[i]>=55 and self.position!='short':
                sl,tp=curr_c-d*0.5,curr_c+d; self.enter('long',curr_c,sl,tp,now)
            elif prev_c>=lower>curr_c and self.df['MA20'].iat[i]<self.df['MA50'].iat[i] and self.df['RSI'].iat[i]<=45 and self.position!='long':
                sl,tp=curr_c+d*0.5,curr_c-d; self.enter('short',curr_c,sl,tp,now)
            # record equity
            self.record_equity(now)
            eq=self.equity_history[-1]
            if now.date()!=self.current_day:
                self.current_day=now.date(); self.daily_high=eq
            self.daily_high=max(self.daily_high,eq); self.all_time_high=max(self.all_time_high,eq)
            if self.daily_high-eq>self.max_daily_dd:
                print(f"Arrêt: drawdown journalier>{self.max_daily_dd}"); return
            if self.all_time_high-eq>self.max_total_dd:
                print(f"Arrêt: drawdown total>{self.max_total_dd}"); return
            if eq-self.capital>=self.objective:
                print(f"Arrêt: objectif {self.objective} atteint"); return
        self.record_equity(self.df.index[-1])

    def report(self):
        df_all = pd.DataFrame(self.trades)
        # include open positions
        open_list=[]
        last_price=self.df['Close'].iloc[-1]
        for leg in self.entries:
            pnl=(leg['size']*((last_price-leg['price'])/leg['price'])) if self.position=='long' else (leg['size']*((leg['price']-last_price)/leg['price']))
            open_list.append({**leg,'exit_time':pd.NaT,'exit_price':last_price,'pnl':pnl,'status':'open'})
        if not df_all.empty: df_all['status']='closed'
        df_report=pd.concat([df_all,pd.DataFrame(open_list)],ignore_index=True)
        df_report.to_csv('trades_log.csv',index=False)
        print('=== STATS BACKTEST ===')
        print(f'Initial capital: {self.capital:,.2f} USD')
        print(f'Final balance  : {self.balance:,.2f} USD')
        eq=np.array(self.equity_history); dd=np.maximum.accumulate(eq)-eq
        print(f'Max drawdown   : {dd.max():,.2f} USD ({dd.max()/self.capital*100:.2f}%)')
        pnl_list=df_report['pnl'].tolist()
        durations=[t['duration'].total_seconds()/3600 if 'duration' in t else (self.df.index[-1]-t['time']).total_seconds()/3600 for t in df_report.to_dict('records')]
        print(f'Trades         : {len(pnl_list)} (closed:{len(self.trades)}, open:{len(open_list)})')
        print(f'Avg duration   : {np.mean(durations):.2f} h')
        if pnl_list:
            print(f'Biggest win    : {max(pnl_list):,.2f} USD')
            print(f'Biggest loss   : {min(pnl_list):,.2f} USD')
        ws=ls=mxws=mxls=0
        for p in pnl_list:
            if p>0: ws+=1; ls=0
            else: ls+=1; ws=0
            mxws=max(mxws,ws); mxls=max(mxls,ls)
        print(f'Win streak     : {mxws} trades')
        print(f'Lose streak    : {mxls} trades')
        # plot
        plt.figure(figsize=(10,6))
        plt.plot(self.times,self.equity_history,label='Equity')
        plt.plot(self.times,self.balance_history,label='Balance',linestyle='--')
        plt.title('Equity vs Balance')
        plt.xlabel('Time');plt.ylabel('USD');plt.legend();plt.grid();plt.show()

if __name__=='__main__':
    args=parse_args()
    bot=BacktestBot(args.risk,args.leverage,args.max_pyramids,args.max_daily_dd,args.max_total_dd,args.objective)
    bot.run()
    bot.report()