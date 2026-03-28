# ==============================
# Trader Behavior Analysis
# ==============================
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set(style="whitegrid")
# ==============================
# 2. Load Data (SAFE LOADING)
# ==============================
# Update paths if needed
sentiment_file = r"C:\Users\Srikanth\OneDrive\Desktop\fear_greed_index.csv"
trades_file = r"C:\Users\Srikanth\OneDrive\Desktop\historical_data.csv"
print("Checking files exist:")
print("Sentiment exists:", os.path.exists(sentiment_file))
print("Trades exists:", os.path.exists(trades_file))
# Load data safely
try:
    sentiment = pd.read_csv(sentiment_file)
    trades = pd.read_csv(trades_file)
except FileNotFoundError as e:
    print("File not found. Check file names and paths.")
    raise e
print("\nSentiment Data Shape:", sentiment.shape)
print("Trades Data Shape:", trades.shape)
# ==============================
# 3. Data Cleaning
# ==============================
# Normalize column names (remove spaces)
sentiment.columns = sentiment.columns.str.strip()
trades.columns = trades.columns.str.strip()
# Convert date columns safely
sentiment['Date'] = pd.to_datetime(sentiment['Date'], errors='coerce')
trades['time'] = pd.to_datetime(trades['time'], errors='coerce')
# Drop invalid rows
sentiment = sentiment.dropna(subset=['Date'])
trades = trades.dropna(subset=['time', 'closedPnL', 'leverage'])
# Extract date
sentiment['date'] = sentiment['Date'].dt.date
required_cols = ['time', 'closedPnL', 'leverage']
trades = trades.dropna(subset=[col for col in required_cols if col in trades.columns])
# ==============================
# 4. Merge Datasets
# ==============================
merged = pd.merge(trades, sentiment, on='date', how='inner')
if merged.empty:
    print("Merged dataset is empty. Check date alignment.")
else:
    print("\nMerged Data Shape:", merged.shape)
# ==============================
# 5. Feature Engineering
# ==============================
merged['profit'] = merged['closedPnL'] > 0
merged['absPnL'] = merged['closedPnL'].abs()
# ==============================
# 6. Analysis
# ==============================
print("\n===== 📊 ANALYSIS RESULTS =====\n")
# Group safely
if 'Classification' not in merged.columns:
    print("❌ 'Classification' column not found. Check dataset.")
    print("Available columns:", merged.columns)
    exit()
grouped = merged.groupby('Classification')
avg_pnl = grouped['closedPnL'].mean()
trade_count = merged['Classification'].value_counts()
avg_leverage = grouped['leverage'].mean()
win_rate = grouped['profit'].mean()
print("Average PnL:\n", avg_pnl)
print("\nTrade Count:\n", trade_count)
print("\nAverage Leverage:\n", avg_leverage)
print("\nWin Rate:\n", win_rate)
# Buy vs Sell (safe)
if 'side' in merged.columns:
    buy_sell = pd.crosstab(merged['Classification'], merged['side'])
    print("\nBuy vs Sell:\n", buy_sell)
# ==============================
# 7. Visualizations
# ==============================
if not merged.empty:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Classification', y='closedPnL', data=merged)
    plt.title("PnL Distribution by Market Sentiment")
    plt.show()
    plt.figure(figsize=(8,5))
    sns.countplot(x='Classification', data=merged)
    plt.title("Trade Frequency by Sentiment")
    plt.show()
    plt.figure(figsize=(8,5))
    sns.barplot(x='Classification', y='leverage', data=merged)
    plt.title("Average Leverage by Sentiment")
    plt.show()
    plt.figure(figsize=(8,5))
    win_rate.plot(kind='bar')
    plt.title("Win Rate by Sentiment")
    plt.ylabel("Win Rate")
    plt.show()
# ==============================
# 8. Insights (SAFE VERSION)
# ==============================
print("\n===== 💡 KEY INSIGHTS =====\n")
fear_pnl = avg_pnl.get('Fear', np.nan)
greed_pnl = avg_pnl.get('Greed', np.nan)
fear_lev = avg_leverage.get('Fear', np.nan)
greed_lev = avg_leverage.get('Greed', np.nan)
fear_win = win_rate.get('Fear', np.nan)
greed_win = win_rate.get('Greed', np.nan)
if not np.isnan(fear_pnl) and not np.isnan(greed_pnl):
    if greed_pnl > fear_pnl:
        print("✔ Traders earn higher profits during GREED periods.")
    else:
        print("✔ Traders earn higher profits during FEAR periods.")

if not np.isnan(fear_lev) and not np.isnan(greed_lev):
    if greed_lev > fear_lev:
        print("✔ Higher leverage is used during GREED (risk-taking behavior).")

if not np.isnan(fear_win) and not np.isnan(greed_win):
    if greed_win > fear_win:
        print("✔ Win rate is higher in GREED markets.")
    else:
        print("✔ Win rate is higher in FEAR markets.")

print("✔ Behavioral differences observed across market sentiment phases.")

# ==============================
# 9. Save Output
# ==============================
if not merged.empty:
    merged.to_csv("merged_analysis.csv", index=False)
    print("\n Analysis Complete! File saved as 'merged_analysis.csv'")
else:
    print("No data to save.")