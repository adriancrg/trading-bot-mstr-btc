# Machine-learning signals for MSTR/BTC and SOL/BTC (hourly data)

A research project that asks: **can a small neural network or a random forest predict short-horizon moves of the
MSTR/BTC ratio and of SOL against BTC well enough to trade them?**

It covers the full workflow: data download, feature engineering, model training, cost-aware backtesting and a
paper-trading bot. **The honest outcome is a working pipeline but no demonstrated trading edge.** The limitations are
listed explicitly below, because understanding why these results should not be trusted is the main thing this project taught me.

> Built with the help of AI coding assistants. Research design, experiments and conclusions are my own.
> This is a learning/research project, not financial advice, and nothing here is evidence of real-world profitability.

## Summary of results

| Experiment | Data | Model | Held-out result (from the saved notebook outputs) | Verdict |
|---|---|---|---|---|
| MSTR/BTC ratio, v4 | 1h bars, ~3,400 aligned hours, 80/20 chronological split | LSTM (1 layer, 32 units, dropout 0.5) | Accuracy 63.5 % test vs 58.3 % train. Long-only: −0.22 cumulative log-return vs −0.92 for buy-and-hold MSTR. Long/short with costs: +42.9 % with 29 position changes | Inconclusive, see limitations |
| SOL vs BTC, first attempt | Binance 1h via ccxt | LSTM | Validation loss ≈ 0.694 (≈ coin flip); backtest −60.1 % | No signal |
| SOL vs BTC, full history | 41,815 hourly rows since June 2021 | LSTM | Validation loss ≈ 0.693; backtest −32.3 % vs market −49.8 % | No signal (loses less than the market only because it is short in a falling market) |
| SOL vs BTC, full history | same | Random Forest (depth 5) | Accuracy 50.5 % test vs 54.9 % train; backtest −88.8 % vs market −51.9 % | No signal |

Re-running the notebooks will give different numbers: Yahoo Finance serves only the last ~730 days of hourly data and
exchange data changes over time. The figures above are those saved in the notebook outputs.

## MSTR/BTC

**Idea.** MSTR (Nasdaq, trades only during market hours) is heavily exposed to Bitcoin (24/7). The MSTR/BTC ratio might
therefore mean-revert on a scale of hours to days.

**Pipeline** (`mstr_btc/`)
1. `01_data_pipeline_and_v1_baseline.ipynb`: download with yfinance, align timestamps to the hour in UTC, keep hours
   where both assets have a price, build features (log returns, ratio, 60-bar rolling mean/std, z-score) and a target
   (ratio at least 0.5 % higher 5 bars later; ~40 % of rows are positive).
2. `02_lstm_v4_training_and_backtest.ipynb`: chronological 80/20 split, `StandardScaler` fitted on the training set only,
   LSTM trained with strong regularisation, then backtests: long-only, and long/short with 0.15 % commission per position
   change and a 20 % APR borrow cost on shorts.
3. `03_alpaca_paper_bot.ipynb`: loads the model, computes the live signal and sends orders to Alpaca's **paper-trading**
   API with a fixed dollar size per trade. No real money was used.

**Data-leakage fix.** The first iteration (notebook 01) fitted its scaler on the whole dataset before splitting, which
leaks information from the test period. Its long/short result should be ignored. Versions 2 and 3 (`archive/`) and the
final v4 fit the scaler on the training data only.

### Why I do not claim this strategy works

* **One short test window in a falling market.** MSTR's cumulative log-return over the test period is −0.92 (roughly a
  60 % price drop, counting only the aligned hours). A model that is short most of the time earns money there without any predictive skill.
* **The model leans short.** Its output probabilities range only from 0.36 to 0.56 (mean 0.46), so it rarely signals
  long. I did not benchmark against a simple "always short" baseline, which is the comparison that matters.
* **Accuracy is probably not informative.** About 60 % of all labels are 0, so always predicting 0 already beats 50 %.
  I did not record the class balance of the test set; validation accuracy being higher than training accuracy is
  consistent with this effect.
* **Thresholds were tuned on the test set.** The long/short thresholds changed (0.60/0.40 to 0.55/0.45) after inspecting
  the prediction range of the model on the held-out data. A proper setup needs a separate validation period.
* **Target and trade do not match.** The model predicts the direction of the MSTR/BTC *ratio* 5 hours ahead, but the
  backtest trades MSTR only, unhedged. This is a directional bet on MSTR, not statistical arbitrage.
* **Simplified time axis.** Rows are hours where both markets are open, so consecutive rows are not always one hour
  apart (overnight and weekend gaps), and the hourly borrow cost is an approximation.
* **No walk-forward validation, no significance test, no confidence intervals.**

## SOL/BTC

**Idea.** Use BTC as a leading indicator for SOL (lead-lag) and trade SOL against it (`sol_btc/`).

**What the notebooks show**
* `analisis_solbtc.ipynb`: on 1,000 hourly bars, correlation 0.99 and an Engle-Granger cointegration p-value of 0.034
  (cointegrated at the 5 % level, but on a short window, so this is fragile evidence).
* `analisis_solana.ipynb`: the same test for SOL against other Solana-ecosystem tokens (JUP, JTO, PYTH, WIF); p-values
  between 0.17 and 0.76, so none qualifies as a cointegrated pair.
* `entrenamiento_solbtc.ipynb` and `full_history_1h/`: LSTMs trained on SOL/BTC features and a Random Forest on 8 engineered
  features (log returns, volume changes, ratio, z-score, RSI, distance to SMA). The LSTM validation loss stays at ≈ 0.693 (= ln 2,
  the loss of a coin flip) and the Random Forest reaches 50.5 % test accuracy. All backtests lose money.

**Conclusion.** With hourly OHLCV-derived features I found no usable predictive signal for SOL. This is consistent with
a highly efficient market where such patterns are arbitraged away quickly, but I did not test why. The Random Forest
notebook plots feature importances, but I did not record the ranking, so I make no claim about which feature mattered most.

## Repository structure

```
mstr_btc/                  final MSTR/BTC pipeline (01 data, 02 model + backtest, 03 paper-trading bot) and model files
sol_btc/                   SOL/BTC analysis and models (first attempt)
sol_btc/full_history_1h/   SOL/BTC with ~5 years of hourly data (LSTM and Random Forest)
archive/                   earlier iterations (v2, v3, a live-signal script), kept for transparency
requirements.txt
.env.example               template for the Alpaca paper-trading keys
```

Notebooks keep their Spanish comments and markdown, and file names inside `sol_btc/` are in Spanish.

## How to run

```bash
pip install -r requirements.txt
# run mstr_btc/01_... then 02_... (CSV files are generated locally and are not committed)
```

For the paper-trading bot, copy `.env.example` to `.env`, add your own Alpaca **paper** keys and run notebook 03.

## Possible next steps

* Benchmark against "always short" and buy-and-hold, and report Sharpe ratio and maximum drawdown.
* Walk-forward validation with a separate validation period for thresholds.
* Trade the actual spread (long one leg, short the other) so that trade and target match.
* Test statistical significance of the results (e.g. bootstrapped confidence intervals).
