import logging
import os
import time
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import requests


BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

RATE_LIMIT_ERROR_CODE = 10006
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1


def _get(url: str, params: dict) -> dict:
    """
    Make GET request to Bybit API with retry logic for rate limit errors.
    Uses exponential backoff: 1s, 2s, 4s delays between retries.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == RATE_LIMIT_ERROR_CODE:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY_SECONDS * (2**attempt)
                    logging.warning(
                        f"Bybit API rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(
                        f"Bybit API rate limit exceeded after {MAX_RETRIES} retries: {data}"
                    )

            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data}")

            return data["result"]

        except (requests.exceptions.RequestException, RuntimeError) as e:
            is_rate_limit_error = (
                isinstance(e, RuntimeError) and "rate limit" in str(e).lower()
            )
            if not is_rate_limit_error and isinstance(e, RuntimeError):
                raise
            if attempt >= MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY_SECONDS * (2**attempt)
            logging.warning(
                f"Bybit API request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed to get response after {MAX_RETRIES} retries")


def get_kline(
    symbol: str,
    interval: Literal[
        "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "M", "W"
    ] = "5",
    *,
    limit: int = 200,
    category: Literal["linear", "inverse", "spot"] = "linear",
) -> list[list[str]]:
    """
    Fetch Bybit v5 kline. Returns raw list data (most-recent first) with fields:
    [start, open, high, low, close, volume, turnover]
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    }
    result = _get(url, params)
    return result["list"]


def klines_to_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    """
    Convert raw kline rows to ascending-indexed DataFrame with numeric columns.
    Columns: time, open, high, low, close, volume, turnover
    """
    if not rows:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume", "turnover"]
        ).astype(
            {
                "time": "datetime64[ns, UTC]",
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float",
                "turnover": "float",
            }
        )

    rows = list(reversed(rows))
    df = pd.DataFrame(
        rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"]
    ).copy()
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = df[col].astype("float64")
    return df


def drop_unclosed_candle(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """
    Remove last row if it is not yet closed relative to now.
    """
    if df.empty:
        return df
    now = datetime.now(timezone.utc)
    last_time = df.iloc[-1]["time"]
    if (now - last_time).total_seconds() < interval_minutes * 60:
        return df.iloc[:-1].copy()
    return df
