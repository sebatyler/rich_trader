from datetime import datetime
from datetime import timezone as dt_timezone

import pytest

from core.market_time import compute_market_time


def test_krx_slot_boundaries_with_explicit_utc_dt():
    # 09:00 KST == 00:00 UTC
    res = compute_market_time(datetime(2026, 1, 5, 0, 0, tzinfo=dt_timezone.utc))
    assert res.krx_slot == "KRX_OPEN"
    assert res.should_run is True

    # 10:00 KST == 01:00 UTC (end-exclusive)
    res = compute_market_time(datetime(2026, 1, 5, 1, 0, tzinfo=dt_timezone.utc))
    assert res.krx_slot is None
    assert res.should_run is False


def test_nyse_slot_winter_est_dst_handling():
    # Winter: EST (UTC-5). 09:30 ET == 14:30 UTC
    res = compute_market_time(datetime(2026, 1, 15, 14, 30, tzinfo=dt_timezone.utc))
    assert res.nyse_slot == "NYSE_OPEN"
    assert res.should_run is True


def test_nyse_slot_summer_edt_dst_handling():
    # Summer: EDT (UTC-4). 09:30 ET == 13:30 UTC
    res = compute_market_time(datetime(2026, 7, 15, 13, 30, tzinfo=dt_timezone.utc))
    assert res.nyse_slot == "NYSE_OPEN"
    assert res.should_run is True


def test_weekend_slot_can_trigger_even_when_markets_closed():
    # 2026-01-03 is Saturday.
    # 15:00 KST == 06:00 UTC, and 01:00 ET (also Saturday in winter).
    res = compute_market_time(datetime(2026, 1, 3, 6, 0, tzinfo=dt_timezone.utc))
    assert res.krx_slot == "KRX_CLOSE"
    assert res.krx_is_open is False
    assert res.nyse_is_open is False
    assert res.markets_closed is True
    assert res.should_run is True
    assert "markets_closed=True" in res.market_time_context


def test_outside_all_slots_should_not_run():
    res = compute_market_time(datetime(2026, 1, 5, 2, 0, tzinfo=dt_timezone.utc))
    assert res.krx_slot is None
    assert res.nyse_slot is None
    assert res.should_run is False


def test_freezegun_usage_example(freezer):
    freezer.move_to("2026-01-05 00:00:00+00:00")
    res = compute_market_time()
    assert res.krx_slot == "KRX_OPEN"
