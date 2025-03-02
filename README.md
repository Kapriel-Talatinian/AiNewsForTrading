# 🤖 AiNewsForTrading

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![cTrader](https://img.shields.io/badge/cTrader-Latest-green.svg)](https://www.ctrader.com/)  
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📌 Overview
AiNewsForTrading is an algorithmic trading solution that leverages real-time financial news analysis powered by GPT-4. It combines a robust C# trading bot with a Python API to deliver actionable trading recommendations based on the latest market news.

## 🚀 Features

### Core Functionality
- Real-time financial news analysis via NewsAPI.
- GPT-4 powered trading recommendations.
- Optimized logging and robust error handling.
- Fast execution for timely trade decisions.

### Supported Components
- **C# Bot:** Executes trades based on recommendations.  
  _Located in `API TRADING/ctraderbot.cs.cs`_
- **Python API:** Retrieves and analyzes financial news using GPT-4.  
  _Located in `API TRADING/api.py`_
- **News Integration:** Uses NewsAPI to fetch the latest financial news.

## 🛠 Tech Stack
- **Backend:** Python (Flask) & C#
- **AI Engine:** OpenAI GPT-4
- **News Integration:** NewsAPI
- **Logging:** Python's logging module
- **Environment Management:** dotenv

## 📋 Prerequisites
- Python 3.10+  
- .NET environment (Visual Studio recommended for C# development)  
- Valid API keys for NewsAPI and OpenAI  
- Internet connection  
- Basic understanding of algorithmic trading  

📬 Contact
📧 Email: kapriel.talatinian@gmail.com
🔗 LinkedIn: Kapriel TALATINIAN
MIT License © 2024 Kapriel TALATINIAN

```mermaid
graph TD;
    A[Start] -->|OnBar Event| B[GetSignalAnalysis]
    B -->|Fetch Signal from API| C[Send API Request]
    C -->|Receive Trading Signal| D[Validate Signal]
    D -->|Check Confidence & Recommendation| E{Valid Signal?}
    E -->|Yes| F[Execute Trade Order]
    E -->|No| G[Ignore Signal]
    F --> H[Order Executed]
    G --> H
    H --> I[End]

    subgraph API Server
        J[Receive API Request] --> K[Fetch News]
        K --> L[Analyze News with GPT-4]
        L --> M[Generate Trading Signal]
        M --> N[Send Response to Bot]
    end
    
    C -->|Request| J
    N -->|Response| D


