import streamlit as st
from streamlit_autorefresh import st_autorefresh
import openai
import os
import json
import pandas as pd
import yfinance as yf
import random
import re
from dotenv import load_dotenv
from datetime import datetime
from time import sleep

# Chargement des clés depuis .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Auto-refresh (toutes les 30s)
st.set_page_config(page_title="News & Trade Simulator", layout="wide")
st_autorefresh(interval=30_000, key="news_trade_autorefresh")

# --- Fonctions ---

def generate_news():
    prompt = (
        "Génère une brève actualité financière en français."
        " Indique explicitement le ticker boursier entre crochets (par ex. [AAPL]) et une description concise."
    )
    resp = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Tu es un générateur d’actualités financières réalistes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=80
    )
    return resp.choices[0].message.content.strip()

def classify_news_with_ticker(news_text):
    system_prompt = (
        "Tu es un assistant financier. Lis la news et renvoie exactement un JSON avec :"
        " 'ticker' (ex: 'AAPL'), 'recommendation' (BUY/SELL/HOLD),"
        " 'confidence' (0.0-1.0), 'reason' (concise)."
    )
    resp = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": news_text}
        ],
        temperature=0.2,
        max_tokens=100
    )
    try:
        result = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        result = {"ticker": None, "recommendation": "HOLD", "confidence": 0.0, "reason": "Erreur JSON"}
    if not result.get("ticker"):
        m = re.search(r"\[([A-Z]{1,5})\]", news_text)
        if m:
            result["ticker"] = m.group(1)
    return result

def fetch_price(ticker, retries=3, delay=1):
    """Récupère le prix via yfinance avec retry sur rate limit. Si échec, génère un prix aléatoire."""
    for _ in range(retries):
        try:
            info = yf.Ticker(ticker).info
            price = info.get("regularMarketPrice")
            if price:
                return price
        except Exception:
            pass
        try:
            df = yf.download(ticker, period="1d")
            if not df.empty:
                return df["Close"].iloc[-1]
        except Exception:
            pass
        sleep(delay)
    # Fallback: prix aléatoire entre 50 et 150
    return random.uniform(50,150)

def simulate_trade(initial_capital, analysis, tp_pct=0.05, sl_pct=0.02, allocation_pct=0.1):
    ticker = analysis.get("ticker")
    rec = analysis.get("recommendation", "HOLD")
    conf = analysis.get("confidence", 0)
    if not ticker:
        return None
    capital = initial_capital
    amount = capital * allocation_pct
    price = fetch_price(ticker)
    entry_price = float(price)
    shares = amount / entry_price if entry_price > 0 else 0
    if rec == "BUY":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    elif rec == "SELL":
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    else:
        tp_price = sl_price = entry_price
    hit_tp = random.random() < conf
    exit_price = tp_price if hit_tp else sl_price
    pnl = (exit_price - entry_price) * shares * (1 if rec == "BUY" else -1)
    new_capital = capital + pnl
    return {
        "timestamp": datetime.utcnow(),
        "ticker": ticker,
        "recommendation": rec,
        "confidence": round(conf, 2),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "pnl": round(pnl, 2),
        "capital": round(new_capital, 2),
        "reason": analysis.get("reason", "")
    }

# --- UI Streamlit ---
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📈 News & Trading Simulator")
news = generate_news()
st.subheader("📰 News générée")
st.write(news)

analysis = classify_news_with_ticker(news)
st.markdown(f"**Ticker:** {analysis.get('ticker', 'N/A')}")
st.markdown(f"**Signal:** {analysis.get('recommendation')} (Confiance: {analysis.get('confidence')})")
st.write(f"_Reason_: {analysis.get('reason')}")

initial_capital = st.session_state.history[-1]["capital"] if st.session_state.history else 10000.0
trade = simulate_trade(initial_capital, analysis)
if trade:
    st.session_state.history.append(trade)
    st.subheader("💼 Dernier Trade")
    st.json(trade)
else:
    st.warning("Pas de trade simulé (ticker invalide ou données manquantes).")

if st.session_state.history:
    st.subheader("📊 Historique des Trades")
    df_hist = pd.DataFrame(st.session_state.history).set_index("timestamp")
    st.dataframe(df_hist)
    st.line_chart(df_hist["capital"])

st.caption(f"Mise à jour: {datetime.utcnow().isoformat()} UTC")
