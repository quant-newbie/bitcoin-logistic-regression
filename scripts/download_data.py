import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str, interval: str, start_time: datetime) -> List[List]:
    """Fetch historical klines from Binance starting at start_time."""
    all_klines: List[List] = []
    current_start = int(start_time.timestamp() * 1000)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000,
        }
        response = requests.get(BINANCE_KLINES_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        all_klines.extend(batch)
        last_open_time = batch[-1][0]
        # Binance returns inclusive ranges; move start to just after last candle.
        current_start = last_open_time + 1

        # Respect API rate limits.
        if len(batch) < 1000:
            break
        time.sleep(0.25)

    return all_klines


def klines_to_dataframe(klines: List[List]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame = pd.DataFrame(klines, columns=columns)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)

    numeric_columns = [c for c in columns if c not in {"open_time", "close_time"}]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    return frame


def save_dataframe(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance kline data to CSV.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--interval", default="1h", help="Kline interval (e.g. 1h, 4h, 1d)")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of historical data to fetch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/btcusdt_1h.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.now(timezone.utc) - timedelta(days=args.days)
    klines = fetch_klines(args.symbol.upper(), args.interval, start_time)
    if not klines:
        raise RuntimeError("No klines returned from Binance. Check symbol/interval combination.")

    frame = klines_to_dataframe(klines)
    save_dataframe(frame, args.output)
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
