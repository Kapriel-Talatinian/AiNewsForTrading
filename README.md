# 🤖 Trading Bot AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![cTrader](https://img.shields.io/badge/cTrader-Latest-green.svg)](https://www.ctrader.com/)  
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📌 Overview

Trading Bot AI is an algorithmic trading solution that leverages real-time financial news analysis powered by GPT-4. It integrates a C# trading bot with a Python API to deliver timely trading recommendations based on market news.

## 🚀 Features

### Core Functionality
- Real-time financial news analysis using NewsAPI.
- GPT-4 powered trading recommendations.
- Optimized logging and robust error handling.
- Fast decision-making with short, efficient responses.

### Supported Components
- **C# Bot:** Executes trades based on analysis.  
  *Located in `API TRADING/ctraderbot.cs.cs`*
- **Python API:** Retrieves news and analyzes it using GPT-4.  
  *Located in `API TRADING/api.py`*
- **Financial News Integration:** Powered by NewsAPI.

## 🛠 Tech Stack
- **Backend:** Python (Flask) & C#
- **AI Engine:** OpenAI GPT-4
- **News Integration:** NewsAPI
- **Logging:** Python's logging module
- **Environment Management:** dotenv

## 📋 Prerequisites
- Python 3.10+  
- .NET environment (Visual Studio recommended for C#)  
- Valid API keys for NewsAPI and OpenAI  
- Basic understanding of algorithmic trading  
- Internet connection

## ⚡ Installation

### Clone Repository
```bash
git clone https://github.com/Kapriel-Talatinian/AiNewsForTrading.git
cd trading-bot-ai
