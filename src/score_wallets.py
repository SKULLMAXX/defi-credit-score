import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os

try:
    with open('./data/transactions.json', 'r') as f:
        data = json.load(f)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'transactions.json' not found in './data/' folder. Please place the file in data/ and try again.")
    exit(1)
except json.JSONDecodeError:
    print("Error: 'transactions.json' is not a valid JSON file.")
    exit(1)

df = pd.json_normalize(data)
print("DataFrame created with", len(df), "transactions.")

def engineer_features(df):
    df['actionData.amount'] = pd.to_numeric(df['actionData.amount'], errors='coerce')
    df['actionData.assetPriceUSD'] = pd.to_numeric(df['actionData.assetPriceUSD'], errors='coerce')
    df['usd_value'] = df['actionData.amount'] * df['actionData.assetPriceUSD']
    
    wallet_features = df.groupby('userWallet').agg({
        'action': ['count', lambda x: (x == 'deposit').sum(), 
                   lambda x: (x == 'borrow').sum(), 
                   lambda x: (x == 'repay').sum(), 
                   lambda x: (x == 'redeemunderlying').sum(), 
                   lambda x: (x == 'liquidationcall').sum()],
        'usd_value': ['sum', 'mean'],
        'timestamp': ['min', 'max'],
        'actionData.assetSymbol': 'nunique'
    }).reset_index()
    
    wallet_features.columns = [
        'userWallet', 'total_txs', 'deposit_count', 'borrow_count', 'repay_count',
        'redeem_count', 'liquidation_count', 'total_usd', 'avg_usd_per_tx',
        'first_tx_time', 'last_tx_time', 'unique_assets'
    ]
    
    wallet_features['total_usd'] = wallet_features['total_usd'].fillna(0)
    wallet_features['avg_usd_per_tx'] = wallet_features['avg_usd_per_tx'].fillna(0)
    
    wallet_features['activity_span_days'] = np.where(
        wallet_features['last_tx_time'] > wallet_features['first_tx_time'],
        (wallet_features['last_tx_time'] - wallet_features['first_tx_time']) / (24 * 3600),
        1
    )
    wallet_features['deposit_to_borrow_ratio'] = wallet_features['deposit_count'] / (wallet_features['borrow_count'] + 1)
    wallet_features['repay_ratio'] = wallet_features['repay_count'] / (wallet_features['borrow_count'] + 1)
    wallet_features['has_liquidation'] = wallet_features['liquidation_count'] > 0
    wallet_features['txs_per_day'] = wallet_features['total_txs'] / (wallet_features['activity_span_days'] + 1e-10)
    
    return wallet_features

features = engineer_features(df)
print("Features engineered for", len(features), "wallets.")

features['synthetic_score'] = (
    0.3 * features['deposit_to_borrow_ratio'] +
    0.3 * features['repay_ratio'] +
    0.2 * features['unique_assets'] -
    0.2 * features['has_liquidation'].astype(int)
)
features['synthetic_score'] = 1000 * (features['synthetic_score'] - features['synthetic_score'].min()) / (features['synthetic_score'].max() - features['synthetic_score'].min())

X = features[['total_txs', 'deposit_count', 'borrow_count', 'repay_count', 'redeem_count',
              'liquidation_count', 'total_usd', 'avg_usd_per_tx', 'activity_span_days',
              'deposit_to_borrow_ratio', 'repay_ratio', 'has_liquidation', 'txs_per_day', 'unique_assets']]
y = features['synthetic_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully.")

features['credit_score'] = model.predict(X_scaled)
features['credit_score'] = 1000 * (features['credit_score'] - features['credit_score'].min()) / (features['credit_score'].max() - features['credit_score'].min())

os.makedirs('./outputs', exist_ok=True)
features[['userWallet', 'credit_score']].to_csv('./outputs/wallet_scores.csv', index=False)
print("Scores saved to ./outputs/wallet_scores.csv")

score_bins = pd.cut(features['credit_score'], bins=range(0, 1100, 100), include_lowest=True)
score_dist = score_bins.value_counts().sort_index()

plt.figure(figsize=(10, 6))
score_dist.plot(kind='bar')
plt.title('Credit Score Distribution')
plt.xlabel('Score Range')
plt.ylabel('Number of Wallets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./outputs/score_distribution.png')
plt.close()
print("Score distribution saved to ./outputs/score_distribution.png")

low_score_wallets = features[features['credit_score'] < 200][['userWallet', 'credit_score', 'deposit_count', 'borrow_count', 'repay_count', 'liquidation_count']]
high_score_wallets = features[features['credit_score'] >= 800][['userWallet', 'credit_score', 'deposit_count', 'borrow_count', 'repay_count', 'liquidation_count']]

os.makedirs('./', exist_ok=True)
with open('./analysis.md', 'w') as f:
    f.write("Analysis of Wallet Credit Scores\n\n")
    f.write("Score Distribution\n![Score Distribution](./outputs/score_distribution.png)\n")
    f.write("Low-Score Wallets (0-200)\n")
    f.write(low_score_wallets.describe().to_string() + "\n")
    f.write("High-Score Wallets (800-1000)\n")
    f.write(high_score_wallets.describe().to_string())
print("Analysis saved to ./analysis.md")