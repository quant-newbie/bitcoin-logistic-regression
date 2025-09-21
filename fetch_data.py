"""Download historical BTCUSDT data from Binance and store as CSV."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Iterable, List
from urllib import request, parse

BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(
    symbol: str,
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
    limit: int = 1000,
) -> List[List]:
    """Fetch historical klines from Binance in batches."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    klines: List[List] = []
    while start_ms < end_ms:
        params = parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
        )
        with request.urlopen(f"{BASE_URL}?{params}") as resp:
            data = resp.read()
        batch = json.loads(data)
        if not batch:
            break
        klines.extend(batch)
        last_open_time = batch[-1][0]
        start_ms = last_open_time + 1
        time.sleep(0.1)
    return klines


def save_to_csv(klines: Iterable[List], output: Path) -> None:
    import csv

    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(klines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument(
        "--start",
        type=lambda s: dt.datetime.fromisoformat(s),
        default=(dt.datetime.utcnow() - dt.timedelta(days=120)).replace(minute=0, second=0, microsecond=0),
    )
    parser.add_argument(
        "--end",
        type=lambda s: dt.datetime.fromisoformat(s),
        default=dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0),
    )
    parser.add_argument("--output", type=Path, default=Path("data/btcusdt_1h.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    klines = fetch_klines(args.symbol, args.interval, args.start, args.end)
    if not klines:
        raise SystemExit("No klines returned. Check symbol/interval/date range.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(klines, args.output)
    print(f"Saved {len(klines)} rows to {args.output}")


if __name__ == "__main__":
    main()
