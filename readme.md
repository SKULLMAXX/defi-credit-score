# Aave V2 Wallet Credit Scoring

## Overview
This project creates a machine learning model to assign credit scores (0-1000) to Aave V2 wallets based on transaction history. Higher scores show reliable usage, while lower scores flag risky behavior.

## Method Chosen
We use a mix of a rule-based synthetic target and a RandomForestRegressor model. The synthetic score weighs deposits (0.3), repayments (0.3), asset diversity (0.2), and penalizes liquidations (-0.2), normalized to 0-1000. The RandomForestRegressor refines this with non-linear patterns from engineered features, balancing interpretability and prediction accuracy.

## Architecture
- **Input**: Reads `./data/transactions.json`.
- **Processing**: Uses pandas for data handling and feature creation.
- **Model**: Employs scikit-learn's RandomForestRegressor with StandardScaler.
- **Output**: Produces `./outputs/wallet_scores.csv`, `./outputs/score_distribution.png`, and `./analysis.md`.
- **Tools**: Relies on pandas, numpy, scikit-learn, matplotlib, and os.

## Processing Flow
1. Loads `./data/transactions.json` into a DataFrame.
2. Engineers features like transaction counts, USD values, activity span, ratios, and liquidation flags.
3. Calculates a synthetic score using the weighted formula.
4. Trains a RandomForestRegressor on scaled features.
5. Predicts and normalizes `credit_score` to 0-1000.
6. Saves scores to CSV, plots distribution, and generates analysis file.

## Requirements
Install these via pip:
- `pip install pandas`
- `pip install numpy`
- `pip install scikit-learn`
- `pip install matplotlib`

## Setup
Place `transactions.json` in `./data/`, then run `python score_wallets.py` from the `src/` directory.

## Validation
Check `wallet_scores.csv` for scores, `score_distribution.png` for distribution, and `analysis.md` for low/high-score stats.

## Extensibility
Add features like time-weighted activity, try Gradient Boosting, or integrate external data.

## Output
- `./outputs/wallet_scores.csv`: Wallet scores.
- `./outputs/score_distribution.png`: Score graph.
- `./analysis.md`: Analysis summary.