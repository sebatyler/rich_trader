from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Optional

from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketTimeResult:
    krx_slot: Optional[str]
    nyse_slot: Optional[str]
    krx_is_open: bool
    nyse_is_open: bool
    markets_closed: bool
    triggered_slots: list[str]
    market_time_context: str
    should_run: bool


def _minutes_since_midnight(dt: datetime) -> int:
    return (dt.hour * 60) + dt.minute


def compute_market_time(now_utc: Optional[datetime] = None) -> MarketTimeResult:
    """Compute KRX/NYSE slot + open state from a UTC timestamp.

    Slot triggering ignores weekday (Option B). Open/closed flags still depend on
    local weekday (<5).
    """

    if now_utc is None:
        now_utc = datetime.now(dt_timezone.utc)

    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")

    now_utc = now_utc.astimezone(dt_timezone.utc)
    kst_now = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
    et_now = now_utc.astimezone(ZoneInfo("America/New_York"))

    kst_minutes = _minutes_since_midnight(kst_now)
    et_minutes = _minutes_since_midnight(et_now)

    krx_slot = None
    if 540 <= kst_minutes < 600:
        krx_slot = "KRX_OPEN"
    elif 720 <= kst_minutes < 780:
        krx_slot = "KRX_MID"
    elif 900 <= kst_minutes < 930:
        krx_slot = "KRX_CLOSE"

    nyse_slot = None
    if 570 <= et_minutes < 630:
        nyse_slot = "NYSE_OPEN"
    elif 720 <= et_minutes < 780:
        nyse_slot = "NYSE_MID"
    elif 900 <= et_minutes < 960:
        nyse_slot = "NYSE_CLOSE"

    krx_is_open = kst_now.weekday() < 5
    nyse_is_open = et_now.weekday() < 5
    markets_closed = (not krx_is_open) and (not nyse_is_open)

    triggered_slots: list[str] = []
    if krx_slot:
        triggered_slots.append(krx_slot)
    if nyse_slot:
        triggered_slots.append(nyse_slot)

    market_time_context = (
        f"kst={kst_now.isoformat()} et={et_now.isoformat()} "
        f"krx_slot={krx_slot or 'NONE'} nyse_slot={nyse_slot or 'NONE'} "
        f"krx_is_open={krx_is_open} nyse_is_open={nyse_is_open} "
        f"triggered_slots={triggered_slots} markets_closed={markets_closed}"
    )

    return MarketTimeResult(
        krx_slot=krx_slot,
        nyse_slot=nyse_slot,
        krx_is_open=krx_is_open,
        nyse_is_open=nyse_is_open,
        markets_closed=markets_closed,
        triggered_slots=triggered_slots,
        market_time_context=market_time_context,
        should_run=bool(triggered_slots),
    )
