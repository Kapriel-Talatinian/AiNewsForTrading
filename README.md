# Trading Bot AI

This repository contains the code for a Trading Bot that leverages financial news analysis powered by GPT-4. The project consists of two main components:
- **C# Bot:** Located in `API TRADING/ctraderbot.cs.cs`
- **Python API:** Located in `API TRADING/api.py`, which retrieves financial news and analyzes it using GPT-4

## Features

- **Financial News Analysis:** Retrieves the latest news using NewsAPI.
- **GPT-4 Integration:** Analyzes news articles to provide trading recommendations.
- **Optimized Logging:** Detailed logs for debugging and performance monitoring.
- **Robust Error Handling:** Improved input validation and error management.
- **Mermaid.js Visualization:** A graphical representation of the project structure.

## Project Structure

```mermaid
graph TD;
    A[Trading Bot AI Repository] --> B[cbot_api];
    B --> C[cBot];
    B --> D[API];
    C --> E[NewsTradingBot.cs];
    D --> F[api.py];
    D --> G[.env];
    A --> H[.gitignore];
    A --> I[README.md];
