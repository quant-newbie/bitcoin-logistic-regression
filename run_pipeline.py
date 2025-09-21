"""Run the Bitcoin logistic regression workflow end-to-end."""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/btcusdt_1h.csv")
DEFAULT_OUTPUT = Path("artifacts/backtest.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to local kline CSV")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--interval", default="1h", help="Kline interval")
    parser.add_argument(
        "--start",
        type=lambda s: dt.datetime.fromisoformat(s),
        default=(dt.datetime.utcnow() - dt.timedelta(days=180)).replace(minute=0, second=0, microsecond=0),
        help="Start datetime (ISO format) for data download when CSV is missing",
    )
    parser.add_argument(
        "--end",
        type=lambda s: dt.datetime.fromisoformat(s),
        default=dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0),
        help="End datetime (ISO format) for data download when CSV is missing",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the backtest equity curve",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not attempt to download data if the CSV is missing",
    )
    return parser.parse_args()


def ensure_data(args: argparse.Namespace) -> Path:
    data_path: Path = args.data_path
    if data_path.exists():
        return data_path
    if args.no_fetch:
        raise FileNotFoundError(f"Data file {data_path} not found and fetching disabled")

    data_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "fetch_data.py",
        "--symbol",
        args.symbol,
        "--interval",
        args.interval,
        "--start",
        args.start.isoformat(),
        "--end",
        args.end.isoformat(),
        "--output",
        str(data_path),
    ]
    subprocess.run(cmd, check=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Expected {data_path} to be created by fetch_data.py")
    return data_path


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.assign(
        open_time=pd.to_datetime(df["open_time"], unit="ms"),
        close_time=pd.to_datetime(df["close_time"], unit="ms"),
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    window_sma = 10
    window_momentum = 3
    window_vol = 10
    window_rsi = 14

    features = df.copy()
    features["momentum"] = features["close"] - features["close"].shift(window_momentum)
    features["sma"] = features["close"].rolling(window_sma).mean()
    features["volatility"] = features["return"].rolling(window_vol).std()

    change = features["close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.rolling(window_rsi).mean()
    avg_loss = loss.rolling(window_rsi).mean()
    rs = avg_gain / avg_loss
    features["rsi"] = 100 - (100 / (1 + rs))

    rolling_vol = features["volume"].rolling(window_vol)
    features["volume_z"] = (features["volume"] - rolling_vol.mean()) / rolling_vol.std()

    feature_cols = ["momentum", "sma", "volatility", "rsi", "volume_z"]
    features = features.dropna(subset=feature_cols).copy()
    features["target"] = (features["close"].shift(-1) > features["close"]).astype(int)
    features = features.dropna(subset=["target"]).copy()
    return features, feature_cols


def train_model(features: pd.DataFrame, feature_cols: list[str]) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(features) * 0.7)
    train = features.iloc[:split_idx]
    test = features.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    test = test.copy()
    test["prob"] = pipeline.predict_proba(X_test)[:, 1]
    test["pred"] = (test["prob"] > 0.5).astype(int)
    return pipeline, train, test


def evaluate(test: pd.DataFrame) -> None:
    accuracy = accuracy_score(test["target"], test["pred"])
    report = classification_report(test["target"], test["pred"])
    print("Test accuracy:", accuracy)
    print(report)


def backtest_and_plot(test: pd.DataFrame, figure_path: Path) -> None:
    test = test.copy()
    test["position"] = np.where(test["prob"] > 0.5, 1, -1)
    test["strategy_return"] = test["position"] * test["return"].shift(-1)
    test["buy_hold"] = test["return"].shift(-1)
    test[["strategy_return", "buy_hold"]] = test[["strategy_return", "buy_hold"]].fillna(0)

    test["strategy_cum"] = (1 + test["strategy_return"]).cumprod()
    test["buy_hold_cum"] = (1 + test["buy_hold"]).cumprod()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test["open_time"], test["strategy_cum"], label="Strategy (LR)")
    ax.plot(test["open_time"], test["buy_hold_cum"], label="Buy & Hold")
    ax.set_title("Backtest Cumulative Returns")
    ax.set_ylabel("Equity Curve")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Saved backtest plot to {figure_path}")


def main() -> None:
    args = parse_args()
    csv_path = ensure_data(args)
    print(f"Using data at {csv_path}")
    df = load_data(csv_path)
    features, feature_cols = engineer_features(df)
    model, train, test = train_model(features, feature_cols)
    evaluate(test)
    backtest_and_plot(test, args.figure_path)


if __name__ == "__main__":
    main()
