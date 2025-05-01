import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import yfinance as yf

# Téléchargement des données avec paramètres corrigés
symbol = "^NDX"
df = yf.download(symbol, start="2022-01-01", end="2024-01-01", auto_adjust=False)

# Correction des noms de colonnes pour yfinance
df.columns = ['_'.join(col).lower().replace(' ', '_') for col in df.columns]

# Sélection des colonnes nécessaires
df = df[['open', 'high', 'low', 'close']].copy()

# Calcul des indicateurs techniques
def calculate_technical_indicators(data):
    # Moyennes mobiles
    data['ma20'] = data['close'].rolling(20).mean()
    data['ma50'] = data['close'].rolling(50).mean()
    
    # ATR (Average True Range)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    data['tr'] = np.maximum.reduce([high_low, high_close, low_close])
    data['atr'] = data['tr'].rolling(14).mean()
    
    # RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    return data.dropna()

df = calculate_technical_indicators(df)

# Paramètres optimisés
params = {
    'capital': 100000,
    'risk_per_trade': 0.02,
    'pitchfork_window': 60,
    'rsi_thresholds': (65, 35),
    'atr_multiplier': 1.5,
    'min_atr': 150
}

# Algorithme de trading
def execute_trading_strategy(data, params):
    balance = params['capital']
    trades = []
    
    for i in range(params['pitchfork_window'], len(data)-1):
        try:
            # Calcul de la régression linéaire
            window = data.iloc[i-params['pitchfork_window']:i]
            x = np.arange(len(window))
            slope, intercept, *_ = linregress(x, window['close'])
            
            # Calcul des niveaux de trading
            midline = slope * (len(window)-1) + intercept
            price_diff = window['close'].iloc[-1] - window['close'].iloc[0]
            upper_band = midline + price_diff
            lower_band = midline - price_diff
            
            current = data.iloc[i]
            next_close = data.iloc[i+1]['close']
            
            # Conditions de trading
            trend_up = current['ma20'] > current['ma50']
            volatility_ok = current['atr'] > params['min_atr']
            rsi_high = current['rsi'] > params['rsi_thresholds'][0]
            rsi_low = current['rsi'] < params['rsi_thresholds'][1]
            
            # Signal long
            if (current['close'] > upper_band and 
                trend_up and 
                volatility_ok and 
                rsi_high):
                
                position_size = (balance * params['risk_per_trade']) / (current['atr'] * params['atr_multiplier'])
                profit = position_size * (next_close - current['close'])
                balance += profit
                trades.append(('long', current.name, current['close']))
                
            # Signal short
            elif (current['close'] < lower_band and 
                  not trend_up and 
                  volatility_ok and 
                  rsi_low):
                
                position_size = (balance * params['risk_per_trade']) / (current['atr'] * params['atr_multiplier'])
                profit = position_size * (current['close'] - next_close)
                balance += profit
                trades.append(('short', current.name, current['close']))
                
        except Exception as e:
            continue
            
    return balance, trades

# Exécution du backtest
final_balance, trades = execute_trading_strategy(df, params)

# Visualisation des résultats
if trades:
    plt.figure(figsize=(14, 7))
    df['close'].plot(label='Prix NDX', alpha=0.6)
    
    long_dates = [t[1] for t in trades if t[0] == 'long']
    long_prices = [t[2] for t in trades if t[0] == 'long']
    plt.scatter(long_dates, long_prices, c='green', marker='^', s=100, label='Achat')
    
    short_dates = [t[1] for t in trades if t[0] == 'short']
    short_prices = [t[2] for t in trades if t[0] == 'short']
    plt.scatter(short_dates, short_prices, c='red', marker='v', s=100, label='Vente')
    
    plt.title(f"Performance finale: ${final_balance:,.2f}")
    plt.legend()
    plt.show()
else:
    print("Aucun trade exécuté - Suggestions :")
    print(f"1. Réduire min_ATR (actuel: {params['min_atr']})")
    print(f"2. Ajuster RSI (actuel: {params['rsi_thresholds']})")
    print(f"3. Données disponibles jusqu'à {df.index[-1].date()}")

# Statistiques finales
print("\nRésumé technique :")
print(f"Capital final: ${final_balance:,.2f}")
print(f"Nombre de trades: {len(trades)}")
print(f"Dernier RSI: {df['rsi'].iloc[-1]:.2f}")
print(f"ATR moyen: {df['atr'].mean():.2f}")
print(f"Ratio MA20/MA50: {df['ma20'].iloc[-1]/df['ma50'].iloc[-1]:.2f}")