# Stock Market Modeling

This repository contains an experiment pipeline centred around `Model_8.1`. The script downloads price data, engineers features, searches hyper‑parameters with Optuna and runs a vectorbt backtest before optionally updating a Google Sheet.

## Setup

1. Install **Python 3.9+**.
2. Clone this repository.
3. (Optional) create and activate a virtual environment.
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. For Google Sheets integration, set the environment variable
    `GOOGLE_APPLICATION_CREDENTIALS` to your service‑account JSON credentials.

## Example usage

Run the full pipeline with:

```bash
python Model_8.1
```

You can also run individual steps using command line flags. Example:

```bash
python Model_8.1 --grid-search --backtest --tickers AAPL,MSFT --start 2020-01-01 --end 2023-01-01
```

If no step flags are provided, all steps run by default.

The script will:

1. Download/cache OHLCV data for the tickers defined in `Model_8.1`.
2. Generate technical indicators and select features via RandomForest.
3. Perform a 10‑trial Optuna Bayesian grid search across RandomForest, XGBoost and LightGBM.
4. Backtest the ensemble strategy with vectorbt and produce `backtest_results.png`
   and `portfolio_values.json`.
5. Append the equity curve to the Google Sheet named in the script (if credentials
   are configured).

Grid‑search results are saved to `grid_search_results.parquet` for later inspection.

## Running tests

Install the dependencies as shown above, then run the unit tests with:

```bash
pytest
```
