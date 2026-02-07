import pytest

from rich.service import collect_crypto_data


def _build_chart(rows: int, start_ts: int, step_sec: int) -> list[dict]:
    chart = []
    for i in range(rows):
        close = 100 + i
        chart.append(
            {
                "timestamp": start_ts + (i * step_sec),
                "open": close - 1,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "target_volume": 10_000 + i,
            }
        )
    return chart


def test_collect_crypto_data_includes_drawdown_and_ema_deviation_fields(
    mocker, settings
):
    settings.DEBUG = True
    daily_chart = _build_chart(rows=120, start_ts=1_700_000_000, step_sec=86_400)
    hourly_chart = _build_chart(rows=120, start_ts=1_700_000_000, step_sec=3_600)

    mocker.patch(
        "rich.service.coinone.get_ticker",
        return_value={
            "best_asks": [{"price": "100"}],
            "best_bids": [{"price": "99"}],
            "high": "110",
            "low": "90",
            "quote_volume": "123456",
            "target_volume": "654321",
            "last": "99.5",
        },
    )

    def _mock_get_candles(symbol: str, interval: str, size: int = 200):
        return {"chart": hourly_chart if interval == "1h" else daily_chart}

    mocker.patch("rich.service.coinone.get_candles", side_effect=_mock_get_candles)
    mock_news = mocker.patch(
        "rich.service.crypto.fetch_news_with_gemini_gap",
        return_value=[],
    )
    mocker.patch(
        "rich.service._compute_coinone_indicators",
        side_effect=[
            {"ema20": 101.0, "ema50": 102.0},
            {"ema20": 120.0, "ema50": 140.0},
        ],
    )

    result = collect_crypto_data(symbol="BTC", start_date="2026-01-01")
    snapshot = result["input_data"]

    current_price = snapshot["current_price"]
    expected_high_30d = max(float(x["high"]) for x in daily_chart[-30:])
    expected_high_90d = max(float(x["high"]) for x in daily_chart[-90:])

    assert snapshot["high_30d"] == expected_high_30d
    assert snapshot["high_90d"] == expected_high_90d
    assert snapshot["drawdown_from_30d_high_pct"] == round(
        ((current_price - expected_high_30d) / expected_high_30d) * 100, 2
    )
    assert snapshot["drawdown_from_90d_high_pct"] == round(
        ((current_price - expected_high_90d) / expected_high_90d) * 100, 2
    )
    assert snapshot["ema20_deviation_pct"] == round(((current_price - 120.0) / 120.0) * 100, 2)
    assert snapshot["ema50_deviation_pct"] == round(((current_price - 140.0) / 140.0) * 100, 2)
    mock_news.assert_not_called()
