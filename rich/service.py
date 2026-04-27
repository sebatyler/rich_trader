import json
import logging
import os
import re
import time
from datetime import timedelta
from decimal import Decimal
from decimal import ROUND_DOWN
from decimal import ROUND_UP
from typing import Optional

import constance
import pandas as pd
from pydantic import BaseModel
from pydantic import Field

from django.conf import settings
from django.db.models import Avg
from django.db.models import Case
from django.db.models import Count
from django.db.models import ExpressionWrapper
from django.db.models import F
from django.db.models import FloatField
from django.db.models import Max
from django.db.models import Min
from django.db.models import When
from django.utils import timezone

import core.coinone as coinone
import core.trading_algorithm as ta
from accounts.models import User
from core import bybit
from core import coinone
from core import crypto
from core import upbit
from core.choices import ExchangeChoices
from core.indicators import atr as calc_atr
from core.indicators import bollinger_bands as calc_bb
from core.indicators import ema as calc_ema
from core.indicators import macd as calc_macd
from core.indicators import rsi as calc_rsi
from core.indicators import volume_ma as calc_volume_ma
from core.llm import invoke_gemini_search
from core.llm import invoke_llm
from core.market_time import compute_market_time
from core.parameter_optimization import create_optimization_payload
from core.parameter_optimization import request_optimized_parameters
from core.telegram import send_message
from core.utils import dict_at
from core.utils import dict_omit
from core.utils import format_currency
from core.utils import format_quantity
from trading.models import AlgorithmParameter
from trading.models import AutoTrading
from trading.models import BybitMechanicalParameter
from trading.models import BybitMechanicalSignal
from trading.models import BybitMechanicalTrade
from trading.models import BybitSignal
from trading.models import Portfolio
from trading.models import Trading
from trading.models import TradingConfig
from trading.models import UpbitTrading

from .models import CryptoListing


MARKET_LIMIT_PRICE_MIN_SPREAD_PCT = Decimal("0.2")
MARKET_LIMIT_PRICE_MIN_EXTRA_PCT = Decimal("0.1")
MARKET_LIMIT_PRICE_MAX_EXTRA_PCT = Decimal("0.3")


def _safe_filename_slug(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "unknown"


def _dump_llm_inputs(
    *,
    name: str,
    prompt: str,
    human_message_template: str,
    template_kwargs: dict,
    user_slug: str,
    template_format: str = "f-string",
    out_dir: str = "tmp/llm_dumps",
) -> dict:
    """Dump LLM prompt + inputs to local files for inspection.

    Writes multiple files to avoid a single massive JSON.
    Returns a dict with created file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    ts = timezone.now().strftime("%Y%m%dT%H%M%S")
    base = os.path.join(
        out_dir, f"{ts}_{_safe_filename_slug(name)}_{_safe_filename_slug(user_slug)}"
    )

    # Render what the LLM will see (best-effort).
    rendered = None
    render_error = None
    if template_format == "f-string":
        try:
            rendered = human_message_template.format(**template_kwargs)
        except Exception as e:
            render_error = f"render_failed: {type(e).__name__}: {e}"

    # Compute rough size stats to help decide what to trim.
    def _byte_len(x) -> int:
        try:
            return len((x or "").encode("utf-8"))
        except Exception:
            return 0

    kw_sizes = []
    for k, v in (template_kwargs or {}).items():
        if isinstance(v, (str, bytes)):
            kw_sizes.append(
                (k, _byte_len(v if isinstance(v, str) else v.decode("utf-8", "ignore")))
            )
        else:
            kw_sizes.append(
                (k, _byte_len(json.dumps(v, ensure_ascii=False, default=str)))
            )
    kw_sizes.sort(key=lambda x: x[1], reverse=True)

    meta = {
        "name": name,
        "timestamp": timezone.now().isoformat(),
        "template_format": template_format,
        "prompt_bytes": _byte_len(prompt),
        "human_template_bytes": _byte_len(human_message_template),
        "human_rendered_bytes": _byte_len(rendered) if rendered is not None else None,
        "render_error": render_error,
        "template_kwargs_keys": list((template_kwargs or {}).keys()),
        "template_kwargs_top10_by_bytes": kw_sizes[:10],
    }

    paths = {
        "base": base,
        "system_prompt": base + ".system.txt",
        "human_template": base + ".human.template.txt",
        "human_rendered": base + ".human.rendered.txt",
        "template_kwargs": base + ".kwargs.json",
        "meta": base + ".meta.json",
    }

    with open(paths["system_prompt"], "w", encoding="utf-8") as f:
        f.write(prompt)

    with open(paths["human_template"], "w", encoding="utf-8") as f:
        f.write(human_message_template)

    if rendered is not None:
        with open(paths["human_rendered"], "w", encoding="utf-8") as f:
            f.write(rendered)

    with open(paths["template_kwargs"], "w", encoding="utf-8") as f:
        json.dump(template_kwargs, f, ensure_ascii=False, indent=2, default=str)

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return paths


def _interval_to_seconds(interval: str) -> int:
    m = re.match(r"^(\d+)([mhd])$", (interval or "").strip().lower())
    if not m:
        raise ValueError(f"Unsupported interval: {interval}")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return n * 60
    if unit == "h":
        return n * 60 * 60
    if unit == "d":
        return n * 24 * 60 * 60
    raise ValueError(f"Unsupported interval unit: {unit}")


def _coinone_chart_to_dataframe(chart: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(chart or [])
    if df.empty:
        return df

    # Coinone chart often returns newest-first; normalize to oldest-first.
    ts_col = None
    for c in ("timestamp", "time"):
        if c in df.columns:
            ts_col = c
            break

    if ts_col:
        ts = pd.to_numeric(df[ts_col], errors="coerce")
        # Heuristic: ms vs sec.
        unit = "ms" if ts.dropna().median() > 1e12 else "s"
        df["time"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")

    for c in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target_volume",
        "quote_volume",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" in df.columns:
        df = df.dropna(subset=["close"])

    if "time" in df.columns:
        df = df.sort_values("time")
    else:
        df = df.iloc[::-1]

    df = df.reset_index(drop=True)
    return df


def _drop_unclosed_candle(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df is None or df.empty or "time" not in df.columns:
        return df
    last_ts = df.iloc[-1].get("time")
    if not isinstance(last_ts, pd.Timestamp) or pd.isna(last_ts):
        return df
    interval_sec = _interval_to_seconds(interval)
    now = pd.Timestamp.now(tz="UTC")
    # Assume chart timestamps represent candle start.
    if last_ts + pd.Timedelta(seconds=interval_sec) > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def _compute_bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    return float(mid.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])


def _compute_coinone_indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    # Prefer target_volume when available.
    if "target_volume" in df.columns:
        volume = df["target_volume"].fillna(0)
    elif "volume" in df.columns:
        volume = df["volume"].fillna(0)
    elif "quote_volume" in df.columns:
        volume = df["quote_volume"].fillna(0)
    else:
        volume = pd.Series([0] * len(df))

    rsi_series = calc_rsi(close, period=14)
    macd_line, signal_line, hist = calc_macd(close, fast=12, slow=26, signal=9)
    ema20 = calc_ema(close, span=20)
    ema50 = calc_ema(close, span=50)
    vma20 = calc_volume_ma(volume, period=20)
    atr14 = calc_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = _compute_bollinger(close, period=20, num_std=2.0)

    last_close = float(close.iloc[-1])
    bb_width = (
        (bb_upper - bb_lower)
        if (bb_upper is not None and bb_lower is not None)
        else None
    )
    bb_pos = None
    if bb_width and bb_width > 0:
        bb_pos = (last_close - bb_lower) / bb_width

    return {
        "close": last_close,
        "rsi14": float(rsi_series.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "bb_mid": float(bb_mid),
        "bb_upper": float(bb_upper),
        "bb_lower": float(bb_lower),
        "bb_pos": float(bb_pos) if bb_pos is not None else None,
        "atr14": float(atr14.iloc[-1]),
        "volume": float(volume.iloc[-1]) if len(volume) else 0.0,
        "volume_ma20": float(vma20.iloc[-1]) if len(vma20) else 0.0,
    }


def _compute_indicators(df: pd.DataFrame, price_col: str = "close"):
    close = df[price_col]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi_series = calc_rsi(close, period=14)
    macd_line, signal_line, hist = calc_macd(close, fast=12, slow=26, signal=9)
    ema20 = calc_ema(close, span=20)
    ema50 = calc_ema(close, span=50)
    vma20 = calc_volume_ma(df["volume"], period=20)
    atr14 = calc_atr(high, low, close, period=14)

    last = df.iloc[-1]
    indicators = {
        "rsi": float(rsi_series.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "volume": float(volume.iloc[-1]),
        "volume_ma20": float(vma20.iloc[-1]),
        "atr": float(atr14.iloc[-1]),
        "close": float(last[price_col]),
    }
    return indicators


def _rule_based_buy_signal(ind_5m: dict, ind_15m: dict) -> bool:
    cond_rsi = ind_5m["rsi"] > 40 and ind_15m["rsi"] >= 45
    cond_macd = ind_5m["macd_hist"] > 0
    cond_trend = (
        ind_5m["close"] > ind_5m["ema20"] and ind_5m["ema20"] >= ind_5m["ema50"]
    )
    cond_vol = ind_5m["volume"] >= ind_5m["volume_ma20"]
    cond_15m_trend = ind_15m["ema20"] >= ind_15m["ema50"]
    return cond_rsi and cond_macd and cond_trend and cond_vol and cond_15m_trend


def _rule_based_scalp_long_signal(ind_3m: dict, ind_5m: dict, ind_15m: dict) -> bool:
    cond_rsi = ind_3m["rsi"] > 45 and ind_5m["rsi"] >= 45
    cond_macd = ind_3m["macd_hist"] > 0
    cond_trend = ind_3m["close"] > ind_3m["ema20"] >= ind_3m["ema50"]
    cond_vol = ind_3m["volume"] >= ind_3m["volume_ma20"]
    cond_htf_trend = (
        ind_5m["ema20"] >= ind_5m["ema50"] and ind_15m["ema20"] >= ind_15m["ema50"]
    )
    return cond_rsi and cond_macd and cond_trend and cond_vol and cond_htf_trend


def _rule_based_scalp_short_signal(ind_3m: dict, ind_5m: dict, ind_15m: dict) -> bool:
    cond_rsi = ind_3m["rsi"] < 55 and ind_5m["rsi"] <= 55
    cond_macd = ind_3m["macd_hist"] < 0
    cond_trend = ind_3m["close"] < ind_3m["ema20"] <= ind_3m["ema50"]
    cond_vol = ind_3m["volume"] >= ind_3m["volume_ma20"]
    cond_htf_trend = (
        ind_5m["ema20"] <= ind_5m["ema50"] and ind_15m["ema20"] <= ind_15m["ema50"]
    )
    return cond_rsi and cond_macd and cond_trend and cond_vol and cond_htf_trend


def _format_telegram_message(
    symbol: str, side: str, tf3: dict, tf5: dict, tf15: dict, decision: dict
) -> str:
    entry = decision.get("entry_price")
    sl = decision.get("stop_loss")
    tp = decision.get("take_profit")
    exp = decision.get("expected_profit_pct")
    conf = decision.get("confidence")
    reason = decision.get("reason", "")
    lev = decision.get("recommended_leverage") or decision.get("leverage")
    few = decision.get("few_minutes_profitable")

    few_txt = "예상 수익 수 분 내 달성 가능" if few else "단기 달성 불확실"
    lines = [
        f"[{side}] Bybit {symbol} 3m 스캘핑 신호",
        f"- 레버리지 {lev}x | 신뢰도 {conf:.2f} | 기대수익 {exp}%",
        f"- {few_txt}",
        f"- 진입 {entry} | 손절 {sl} | 익절 {tp}",
        f"- 3m 종가 {tf3['close']:.4f} | RSI {tf3['rsi']:.1f} | MACD hist {tf3['macd_hist']:.3f}",
        f"- 5m RSI {tf5['rsi']:.1f} | 15m RSI {tf15['rsi']:.1f}",
        (f"- 사유: {reason}" if reason else None),
    ]
    return "\n".join([str(x) for x in lines if x])


def scan_bybit_signals():
    # Load all configs that enabled Bybit alerts and aggregate target symbols
    configs = list(TradingConfig.objects.filter(bybit_alert_enabled=True))
    symbols_set = set()
    for cfg in configs:
        coins = getattr(cfg, "bybit_target_coins", None) or []
        for coin in coins:
            symbols_set.add(coin)

    results = []

    for symbol in sorted(symbols_set):
        # fetch klines
        rows_3 = bybit.get_kline(symbol, interval="3", limit=200, category="linear")
        rows_5 = bybit.get_kline(symbol, interval="5", limit=200, category="linear")
        rows_15 = bybit.get_kline(symbol, interval="15", limit=200, category="linear")
        df3 = bybit.klines_to_dataframe(rows_3)
        df5 = bybit.klines_to_dataframe(rows_5)
        df15 = bybit.klines_to_dataframe(rows_15)

        # ensure closed candle only
        df3 = bybit.drop_unclosed_candle(df3, interval_minutes=3)
        df5 = bybit.drop_unclosed_candle(df5, interval_minutes=5)
        df15 = bybit.drop_unclosed_candle(df15, interval_minutes=15)
        if len(df3) < 50 or len(df5) < 50 or len(df15) < 50:
            continue

        ind3 = _compute_indicators(df3)
        ind5 = _compute_indicators(df5)
        ind15 = _compute_indicators(df15)
        should_long = _rule_based_scalp_long_signal(ind3, ind5, ind15)
        should_short = _rule_based_scalp_short_signal(ind3, ind5, ind15)

        # LLM decision
        payload = {
            "symbol": symbol,
            "timeframes": ["3m", "5m", "15m"],
            "indicators": {
                "rsi_3m": ind3["rsi"],
                "rsi_5m": ind5["rsi"],
                "rsi_15m": ind15["rsi"],
                "macd_3m": {
                    "macd": ind3["macd"],
                    "signal": ind3["macd_signal"],
                    "hist": ind3["macd_hist"],
                },
                "macd_5m": {
                    "macd": ind5["macd"],
                    "signal": ind5["macd_signal"],
                    "hist": ind5["macd_hist"],
                },
                "ema_3m": {"ema20": ind3["ema20"], "ema50": ind3["ema50"]},
                "ema_5m": {"ema20": ind5["ema20"], "ema50": ind5["ema50"]},
                "ema_15m": {"ema20": ind15["ema20"], "ema50": ind15["ema50"]},
                "vol_ma_3m": {"vol": ind3["volume"], "vol_ma20": ind3["volume_ma20"]},
                "atr_3m": ind3["atr"],
            },
            "last_closed_candle": {
                "time": df3.iloc[-1]["time"].isoformat(),
                "close": ind3["close"],
            },
            "fees": {"derivatives_taker": 0.00055, "derivatives_maker": 0.0002},
            "rule_based": {"long": should_long, "short": should_short},
        }

        class BybitDecision(BaseModel):
            trade_signal: bool = Field(..., description="Whether to enter a trade now")
            side: str = Field(..., description="Trade direction: LONG or SHORT")
            confidence: float = Field(..., description="The confidence 0-1")
            reason: str = Field(..., description="The reason in Korean (<=2 sentences)")
            entry_price: float = Field(..., description="The entry price")
            stop_loss: float = Field(..., description="The stop loss")
            take_profit: float = Field(
                ..., description="The take profit reachable in minutes"
            )
            expected_profit_pct: float = Field(
                ..., description="Net expected profit % at 1x, fees included"
            )
            recommended_leverage: int = Field(
                ..., description="Leverage (e.g., 10/25/50)"
            )
            few_minutes_profitable: bool = Field(
                ..., description="Likely to realize within next 1-3 3m candles"
            )

        system = (
            "You are a trading assistant for Bybit USDT perpetual futures (cross). "
            "Decide if a SHORT-TERM SCALP entry is reasonable NOW for either LONG or SHORT based ONLY on the provided indicators and last closed candles. "
            "Optimize for scalping: prefer setups with momentum realizable within the next 1-3 candles on 3m. "
            "Be conservative if signals are mixed or volume confirmation is weak. "
            "Include derivatives trading fees in all profit expectations. Taker fee = 0.055% per leg, Maker fee = 0.020% per leg. Assume TAKER in and out. "
            "Compute expected_profit_pct exactly as: ((take_profit / entry_price) - 1 - entry_fee - exit_fee) * 100 for LONG and the analogous formula for SHORT using (entry_price / take_profit - 1). Entry/exit fees equal taker fee by default. "
            "Pick recommended_leverage from {10,25,50} based on stop distance and volatility; smaller stops and stronger momentum allow higher leverage, else prefer lower. "
            "If expected_profit_pct < 0.1, set trade_signal=false regardless of other signals and briefly explain why in Korean (<= 2 sentences). "
            "Only set trade_signal=true when risk-reward >= 1.5 (net of fees) and the setup is likely realizable within the next 1-3 3m candles. "
            "Respond strictly in JSON matching the provided schema. The 'reason' must be written in Korean (<= 2 sentences)."
        )
        content = json.dumps(payload, ensure_ascii=False)
        decision_obj = invoke_llm(
            system,
            content,
            model=BybitDecision,
            structured_output=True,
            template_format="jinja2",
        )
        decision = decision_obj.model_dump()

        # persist snapshot for 3m timeframe (scalping)
        BybitSignal.objects.update_or_create(
            symbol=symbol,
            timeframe="3m",
            last_candle_time=df3.iloc[-1]["time"],
            defaults=dict(
                close_price=ind3["close"],
                rsi=ind3["rsi"],
                macd=ind3["macd"],
                macd_signal=ind3["macd_signal"],
                macd_hist=ind3["macd_hist"],
                ema20=ind3["ema20"],
                ema50=ind3["ema50"],
                volume=ind3["volume"],
                volume_ma20=ind3["volume_ma20"],
                atr=ind3["atr"],
                trade_signal=bool(decision.get("trade_signal", False)),
                side=decision.get("side"),
                confidence=decision.get("confidence"),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                expected_profit_pct=decision.get("expected_profit_pct"),
                recommended_leverage=decision.get("recommended_leverage"),
                few_minutes_profitable=bool(decision.get("few_minutes_profitable")),
                decision=decision,
            ),
        )

        # Notify per-config: only when LLM says trade and expected_profit_pct >= 0.1
        if decision.get("error"):
            logging.exception(f"Bybit decision error for {symbol}: {decision['error']}")
        else:
            exp = decision.get("expected_profit_pct")
            should_notify = (
                bool(decision.get("trade_signal"))
                and isinstance(exp, (int, float))
                and exp >= 0.1
            )
            if should_notify:
                text = _format_telegram_message(
                    symbol, decision.get("side", "LONG"), ind3, ind5, ind15, decision
                )
                for cfg in configs:
                    # double-check enabled and membership
                    if cfg.bybit_alert_enabled and symbol in (
                        cfg.bybit_target_coins or []
                    ):
                        chat_id = cfg.telegram_chat_id
                        if chat_id:
                            try:
                                send_message(text, chat_id=chat_id, is_markdown=False)
                            except Exception:
                                logging.exception(
                                    f"Failed to send Telegram message for {symbol} to chat_id={chat_id}"
                                )

        results.append(
            {
                "symbol": symbol,
                "should_long": should_long,
                "should_short": should_short,
                "decision": decision,
            }
        )

    return results


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class Recommendation(BaseStrippedModel):
    action: str = Field(..., description="The action to take (BUY or SELL)")
    symbol: str = Field(..., description="The symbol of the cryptocurrency")
    amount: Optional[int] = Field(
        default=None, description="The amount of the cryptocurrency to buy in KRW"
    )
    quantity: Optional[float] = Field(
        default=None, description="The quantity of the cryptocurrency to sell"
    )
    limit_price: Optional[float] = Field(
        default=None, description="The limit price for the order"
    )
    reason: str = Field(..., description="The reason for the recommendation")


class MultiCryptoRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The analysis scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    recommendations: list[Recommendation] = Field(
        ..., description="List of recommended cryptocurrency trades"
    )


def collect_crypto_data(
    symbol: str, start_date: str, news_count: int = 10, from_upbit: bool = False
):
    """특정 암호화폐의 모든 관련 데이터를 수집합니다."""
    # Default optional fields for callers.
    crypto_data_csv = ""
    network_stats_csv = ""
    ta_1h = {}
    ta_1d = {}
    closes_1h_csv = ""
    closes_1d_csv = ""

    if from_upbit:
        # Upbit/rebalance flow keeps richer fundamentals/history.
        tickers = upbit.get_ticker(symbol)
        ticker = tickers[0]
        crypto_price = ticker["trade_price"]

        crypto_data = crypto.get_quotes(symbol)
        input_data = dict(
            ticker,
            circulating_supply=crypto_data["circulating_supply"],
            max_supply=crypto_data["max_supply"],
            total_supply=crypto_data["total_supply"],
            **crypto_data["quote"]["KRW"],
            current_price=crypto_price,
        )

        historical_data = crypto.get_historical_data(symbol, "KRW", 30)
        df = pd.DataFrame(historical_data)
        df = df.drop(columns=["conversionType", "conversionSymbol"])
        crypto_data_csv = df.to_csv(index=False)

        if symbol == "BTC":
            network_stats = crypto.get_network_stats()
            df = pd.DataFrame(network_stats, index=[0])
            network_stats_csv = df.to_csv(index=False)
    else:
        # Coinone auto trading: indicator-centric payload to reduce tokens.
        ticker = coinone.get_ticker(symbol)
        best_asks = ticker.get("best_asks") or []
        best_bids = ticker.get("best_bids") or []
        best_ask = float((best_asks[0] or {}).get("price") or 0) if best_asks else 0.0
        best_bid = float((best_bids[0] or {}).get("price") or 0) if best_bids else 0.0
        crypto_price = (
            (best_ask + best_bid) / 2
            if best_ask and best_bid
            else float(ticker.get("last") or 0)
        )

        # Candles for indicators
        candles_1h = coinone.get_candles(symbol, "1h", size=200)
        df_1h = _coinone_chart_to_dataframe((candles_1h or {}).get("chart") or [])
        df_1h = _drop_unclosed_candle(df_1h, "1h")

        candles_1d = coinone.get_candles(symbol, "1d", size=180)
        df_1d = _coinone_chart_to_dataframe((candles_1d or {}).get("chart") or [])
        df_1d = _drop_unclosed_candle(df_1d, "1d")

        if len(df_1h) >= 60:
            ta_1h = _compute_coinone_indicators(df_1h)
            closes_1h_csv = df_1h[["time", "close"]].tail(12).to_csv(index=False)
        if len(df_1d) >= 60:
            ta_1d = _compute_coinone_indicators(df_1d)
            closes_1d_csv = df_1d[["time", "close"]].tail(14).to_csv(index=False)

        # Compute basic returns from daily series.
        ret_7d = None
        ret_30d = None
        if len(df_1d) >= 8:
            ret_7d = (
                float(df_1d["close"].iloc[-1]) / float(df_1d["close"].iloc[-8]) - 1
            ) * 100
        if len(df_1d) >= 31:
            ret_30d = (
                float(df_1d["close"].iloc[-1]) / float(df_1d["close"].iloc[-31]) - 1
            ) * 100

        # Drawdown from 30d/90d highs and deviation from key moving averages.
        high_30d = None
        high_90d = None
        drawdown_from_30d_high_pct = None
        drawdown_from_90d_high_pct = None
        ema20_deviation_pct = None
        ema50_deviation_pct = None

        if "high" in df_1d.columns and len(df_1d) >= 30:
            high_30d = float(df_1d["high"].tail(30).max())
        if "high" in df_1d.columns and len(df_1d) >= 90:
            high_90d = float(df_1d["high"].tail(90).max())

        if high_30d and crypto_price:
            drawdown_from_30d_high_pct = round(
                ((float(crypto_price) - high_30d) / high_30d) * 100,
                2,
            )
        if high_90d and crypto_price:
            drawdown_from_90d_high_pct = round(
                ((float(crypto_price) - high_90d) / high_90d) * 100,
                2,
            )

        ema20 = (ta_1d or {}).get("ema20")
        ema50 = (ta_1d or {}).get("ema50")
        if ema20:
            ema20_deviation_pct = round(
                ((float(crypto_price) - float(ema20)) / float(ema20)) * 100,
                2,
            )
        if ema50:
            ema50_deviation_pct = round(
                ((float(crypto_price) - float(ema50)) / float(ema50)) * 100,
                2,
            )

        spread_pct = None
        if best_ask and best_bid:
            spread_pct = ((best_ask / best_bid) - 1) * 100

        input_data = {
            "symbol": symbol,
            "current_price": float(crypto_price or 0),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_pct": spread_pct,
            "high_24h": float(ticker.get("high") or 0),
            "low_24h": float(ticker.get("low") or 0),
            "quote_volume_24h": float(ticker.get("quote_volume") or 0),
            "target_volume_24h": float(ticker.get("target_volume") or 0),
            "ret_7d_pct": ret_7d,
            "ret_30d_pct": ret_30d,
            "high_30d": high_30d,
            "high_90d": high_90d,
            "drawdown_from_30d_high_pct": drawdown_from_30d_high_pct,
            "drawdown_from_90d_high_pct": drawdown_from_90d_high_pct,
            "ema20_deviation_pct": ema20_deviation_pct,
            "ema50_deviation_pct": ema50_deviation_pct,
        }

    # 뉴스 데이터
    if settings.DEBUG:
        crypto_news_csv = ""
    else:
        # Use API + Gemini gap backfill to improve freshness
        crypto_news = crypto.fetch_news_with_gemini_gap(start_date, symbol, news_count)
        df = (
            pd.DataFrame(crypto_news)
            if crypto_news
            else pd.DataFrame(
                columns=["source", "title", "description", "publishedAt", "content"]
            )
        )
        if not df.empty:
            # Ensure consistent columns and source name normalization
            if "source" in df.columns:
                df["source"] = df["source"].apply(
                    lambda x: (x or {}).get("name") if isinstance(x, dict) else x
                )
            expected_cols = ["source", "title", "description", "publishedAt", "content"]
            missing = [c for c in expected_cols if c not in df.columns]
            for c in missing:
                df[c] = None
            df = df[expected_cols]
        crypto_news_csv = df.to_csv(index=False)

    return {
        "symbol": symbol,
        "input_data": input_data,
        "crypto_data_csv": crypto_data_csv,
        "network_stats_csv": network_stats_csv,
        "crypto_news_csv": crypto_news_csv,
        "ta_1h": ta_1h,
        "ta_1d": ta_1d,
        "closes_1h_csv": closes_1h_csv,
        "closes_1d_csv": closes_1d_csv,
    }


def get_multi_recommendation(
    crypto_data_list: list[dict],
    indices_csv: str,
    balances: dict[str, dict],
    total_coin_value: int,
    markets: dict[str, dict],
    recent_trades_csv: str,
    trading_config: TradingConfig,
    with_fallback: bool = False,
    market_time_context: Optional[str] = None,
) -> MultiCryptoRecommendation:
    """LLM을 사용하여 암호화폐 투자 추천을 받습니다."""
    # 각 코인별 데이터를 하나의 문자열로 조합
    data_descriptions = []
    for data in crypto_data_list:
        symbol = data["symbol"]
        description = f"""
=== {symbol} Data ===
User's current balance data in KRW in JSON
```json
{symbol}_balance_json
```
Market data in JSON
```json
{symbol}_market_json
```
Market snapshot in JSON
```json
{symbol}_snapshot_json
```
Technical indicators (computed from Coinone candles)
1h timeframe indicators in JSON
```json
{symbol}_ta_1h_json
```
1d timeframe indicators in JSON
```json
{symbol}_ta_1d_json
```
Recent closes (minimal raw data)
1h closes (last 12 rows) in CSV
```csv
{symbol}_closes_1h_csv
```
1d closes (last 14 rows) in CSV
```csv
{symbol}_closes_1d_csv
```"""

        description += f"""
News in CSV
```csv
{symbol}_crypto_news_csv
```"""
        description = re.sub(
            rf"^({symbol}_(snapshot_json|ta_1h_json|ta_1d_json|closes_1h_csv|closes_1d_csv|crypto_news_csv|balance_json|market_json))",
            r"{\1}",
            description,
            flags=re.MULTILINE,
        )
        data_descriptions.append(description)

    all_data = """
=== Market Indices ===
Indices data in USD in CSV
```csv
{indices_csv}
```

=== Recent Trades ===
Recent trades in KRW in CSV
```csv
{recent_trades_csv}
```""".strip()
    all_data += "\n\n" + "\n".join(data_descriptions)

    # 각 코인별 데이터를 개별 변수로 전달하기 위한 kwargs 구성
    kwargs = {
        "indices_csv": indices_csv,
        "recent_trades_csv": recent_trades_csv,
    }

    # 각 코인별로 데이터 변수 추가
    for data in crypto_data_list:
        symbol = data["symbol"]
        balance = balances.get(symbol, {})
        market = markets.get(symbol, {})
        # 매도 시 필요한 정보만 추출
        market = {k: v for k, v in market.items() if "qty" in k}
        kwargs.update(
            {
                f"{symbol}_balance_json": json.dumps(balance),
                f"{symbol}_snapshot_json": json.dumps(
                    data["input_data"], ensure_ascii=False
                ),
                f"{symbol}_market_json": json.dumps(market),
                f"{symbol}_ta_1h_json": json.dumps(
                    data.get("ta_1h") or {}, ensure_ascii=False
                ),
                f"{symbol}_ta_1d_json": json.dumps(
                    data.get("ta_1d") or {}, ensure_ascii=False
                ),
                f"{symbol}_closes_1h_csv": data.get("closes_1h_csv") or "",
                f"{symbol}_closes_1d_csv": data.get("closes_1d_csv") or "",
                f"{symbol}_crypto_news_csv": data["crypto_news_csv"],
            }
        )

    def _fmt_pct(value):
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return "N/A"

    opportunity_lines = []
    for data in crypto_data_list:
        symbol = data["symbol"]
        snapshot = data.get("input_data") or {}
        opportunity_lines.append(
            (
                f"- {symbol}: dd30={_fmt_pct(snapshot.get('drawdown_from_30d_high_pct'))}, "
                f"dd90={_fmt_pct(snapshot.get('drawdown_from_90d_high_pct'))}, "
                f"ema20_dev={_fmt_pct(snapshot.get('ema20_deviation_pct'))}, "
                f"ema50_dev={_fmt_pct(snapshot.get('ema50_deviation_pct'))}"
            )
        )
    opportunity_snapshot = (
        "\n".join(opportunity_lines) if opportunity_lines else "- N/A"
    )

    krw_balance = int(float(balances["KRW"]["available"] or 0))
    prompt = f"""You are a short-term and swing crypto trading advisor invoked hourly at :15. Perform heavy evaluation ONLY during session-slot windows (KRX/KST OPEN/MID/CLOSE and NYSE/ET OPEN/MID/CLOSE; up to 6 per day). At each slot trigger, analyze the CURRENT MARKET CONDITIONS and recommend the BEST POSSIBLE TRADES based on available data. This account is dedicated to short-term/swing opportunities, not long-term buy-and-hold portfolio management. You have access to:
 - Market snapshot (price/spread/returns), account balances, order constraints
 - Technical indicators computed from Coinone candles (1h, 1d)
 - Minimal raw closes (1h/1d) for context only
  - News (raw)
  - Recent trading history in CSV format (use this to learn from past decisions and patterns)
 - KRW balance: {krw_balance:,} KRW
 - Total coin value: {total_coin_value:,} KRW
 - Total portfolio value: {total_coin_value + krw_balance:,} KRW
 - Min trade: {trading_config.min_trade_amount:,} KRW, step: {trading_config.step_amount:,} KRW
 - Drawdown/EMA deviation snapshot:
{opportunity_snapshot}

CRITICAL CONTEXT - EVALUATION AT THIS MOMENT:
- Invoked hourly at :15; heavy evaluation occurs only during session-slot windows (up to 6 per day)
- Market time context: {market_time_context or "N/A"}
- If markets_closed=true, treat equity indices as stale/unchanged (previous close) and do not over-weight them
- Your goal: Assess the CURRENT SITUATION and recommend the OPTIMAL trades RIGHT NOW
- You are NOT required to make trades every cycle - only recommend when opportunities are genuinely attractive
- Use ALL provided data (prices, indicators, news, recent trades) to make informed decisions
- Consider the cumulative impact of recent trades on your portfolio and strategy
- Focus on capturing evidence-backed short-term opportunities while preserving risk discipline
- Do not force trades, but avoid defaulting to passive waiting when short-term opportunity signals are clear

Key Rules (CRITICAL - FOLLOW EXACTLY):
1) Trade Recommendation Count Rules:
   - Recommend {trading_config.min_coins} to {trading_config.max_coins} trades if good opportunities exist, OR 0 if current market conditions don't warrant action
   - NEVER exceed {trading_config.max_coins} trades
   - NEVER recommend both BUY and SELL for the same coin
   - Each coin can appear only once in recommendations
   - Analyze recent trading history: If a coin was traded recently, evaluate whether current conditions justify another trade or if waiting is better
   - Quality over quantity: Only recommend trades when they genuinely improve portfolio position or manage risk effectively

2) BUY Constraints (Optimal Entry Points):
   - amount ≥ {trading_config.min_trade_amount}, multiple of {trading_config.step_amount}
   - Single BUY ≤ 30% of available KRW, total BUY ≤ 50% of KRW
   - Execute BUY as MARKET orders only (no limit/post-only)
   - Recommend BUY when current market conditions suggest favorable entry:
     a) Strong upward momentum indicators (RSI, MACD, price action alignment) OR oversold rebound setup (sharp prior drop + early 1h rebound signals)
     b) Positive news/sentiment OR technical breakout confirmation OR evidence of short-term mean reversion after oversold conditions
     c) Price is at reasonable levels (not FOMO buying at recent highs)
     d) Volume confirms genuine interest
     e) Expected price appreciation justifies fees (≥ 0.1% after 0.04% round-trip fees)
   - Consider recent trading history: If coin was sold recently, compare current price vs recent sell prices. Re-entry can be valid at meaningfully lower prices when short-term setup improves; be more cautious when price is at/above recent sell levels
   - If coin was bought recently, evaluate if additional buying improves position or if holding is better

3) SELL Constraints (Optimal Exit Points & Risk Management):
   - quantity must respect exchange increments (qty_unit) and min_qty~max_qty range
   - Consider partial selling if large holdings, to manage risk and slippage
   - Execute SELL as MARKET orders only; set limit_price: null
   - Recommend SELL when current conditions suggest it's optimal:
     a) Downward trend confirmed (RSI, MACD, price action showing weakness)
     b) Risk mitigation needed (portfolio risk too high, or stop-loss considerations)
     c) Take profit opportunity (sufficient profit after fees, or strong resistance reached)
     d) Negative news/sentiment shift that could cause immediate price drop
   - Evaluate current holdings: If holding at a loss, assess whether cutting losses now is better than waiting
   - For profitable positions: Consider if taking profits now is optimal or if holding for larger gains is better
   - Use recent trading history to avoid emotional decisions (don't sell just because you sold before)

4) Fees & Profit Considerations:
    - Fee: 0.02% each trade (0.04% round-trip = buy + sell)
    - Consider fees when evaluating trade profitability: Price needs to move ≥ 0.5% to justify trade (covers 0.04% fees + 0.46% minimum profit)
    - Each trade should have sufficient expected profit potential to justify fees
    - Be aware of cumulative fees from recent trades - factor this into decision-making
    - Avoid trades where expected profit is marginal compared to fees
    - Prioritize higher conviction opportunities over marginal gains

5) Risk & Volatility Management:
   - Avoid risking >2~3% of total portfolio on a single trade
   - Adjust position sizes based on volatility - higher volatility suggests smaller positions
   - Factor in recent news/sentiment when evaluating current opportunities
   - Analyze recent trading performance from CSV data:
     * Learn from past trades: Which coins performed well? Which didn't?
     * Use this information to inform current decisions, but don't let past performance blind you to current opportunities
   - Portfolio health check: Assess current portfolio risk and adjust recommendations accordingly
   - Use volatility indicators (ATR) to inform position sizing decisions

6) Portfolio Balance (KRW Ratio):
   - After ALL recommended BUY/SELL are done, evaluate if KRW ratio is appropriate (target: 10%~50%)
   - If KRW ratio is above target, allow selective entries when short-term rebound opportunities are clear
   - In extremely volatile/uncertain conditions, maintaining a higher KRW ratio is still acceptable for risk control

7) Recent Trading Analysis (Learn from History):
   - Review recent trades from CSV data to inform current decisions:
     * Analyze win rate and profit/loss patterns for each coin
     * Identify which coins have been profitable vs unprofitable
     * Use this information to evaluate current opportunities, but don't let it prevent you from recognizing new patterns
   - Trade frequency consideration:
     * Evaluate if recent trading frequency is appropriate given market conditions
     * Consider whether current market conditions justify more or fewer trades
   - Learn from past decisions:
     * What worked well? What didn't?
     * How can you apply these lessons to current market conditions?
   - Pattern recognition:
     * Identify patterns in successful vs unsuccessful trades
     * Use patterns to inform decisions, but remain flexible to changing conditions

8) Current Market Evaluation:
   - Consider current market conditions: time of day, market hours (Asian/European/US), volatility
   - Evaluate whether NOW is a good time to trade or if waiting is better
   - Don't feel pressured to trade - sometimes the best decision is to do nothing
   - Assess if market conditions are clear enough to make confident decisions
   - Consider the timeframe: session-slot windows (short-to-medium term)

9) Spread / Liquidity Safety (STRICT):
   - Use spread_pct from Market snapshot.
   - If spread_pct is null OR spread_pct >= 1.0%, do NOT recommend BUY for that coin.
   - If spread_pct >= 2.0%, do NOT recommend any trade (BUY/SELL) for that coin.
   - Prefer skipping a trade over trading illiquid/abnormal orderbooks.

Output must be valid YAML with these sections:
```yaml
scratchpad: |
  [현재 시장 상황과 최근 거래 분석 (한국어). 핵심 포인트만 3-4줄로 작성]
  - 제공된 데이터(가격, 지표, 뉴스)를 기반으로 한 현재 시장 평가
  - 최근 거래 패턴 분석 및 교훈

reasoning: |
  [현재 시점에서의 최적 매매 전략 설명 (한국어). 핵심 포인트만 3-4줄로 작성]
  - 각 추천의 근거 (현재 시장 조건, 예상 수익성, 리스크 평가)
  - 거래를 하지 않는 경우, 그 이유 설명

recommendations:
  - action: "BUY"    # or "SELL"
    symbol: "BTC"
    amount: 500000   # (int or null) for BUY only
    quantity: null   # (float or null) for SELL only
    limit_price: null  # (must be null for SELL; MARKET execution only)
    reason: "현재 시점에서 이 거래를 추천하는 핵심 사유 1-2줄로 작성"
```

Rules:
1. Strictly follow the YAML structure above
2. scratchpad and reasoning MUST use multiline string format with | operator and consistent indentation
3. CRITICAL: After the | operator, EVERY line of content MUST start with exactly 2 spaces indentation
4. Example of CORRECT format:
   scratchpad: |
     First line with 2 spaces indent
     Second line with 2 spaces indent
5. Example of INCORRECT format (will cause parse error):
   scratchpad: |
   First line without indent (WRONG!)
6. Keep total length of scratchpad + reasoning < 2000 chars
4. Each line should be a complete, meaningful statement
5. Use simple, clear Korean language
6. No repetition between sections
7. Double-check that recommendations follow all trade recommendation count and constraint rules
8. No extra fields. No extra lines outside the YAML
9. If current market conditions don't warrant trades, recommend 0 trades (empty recommendations list) and explain why in reasoning
10. Base all recommendations on CURRENT market conditions and data provided, not on pressure to trade
    """
    if settings.DEBUG:
        # Dump-only mode for local inspection (superuser only).
        if trading_config.user.is_superuser:
            user_slug = (trading_config.user.email or "superuser").split("@")[0]
            _dump_llm_inputs(
                name="coinone-auto-trading-4h",
                prompt=prompt,
                human_message_template=all_data,
                template_kwargs=kwargs,
                user_slug=user_slug,
                template_format="f-string",
            )

        # Do NOT invoke the LLM in DEBUG.
        return MultiCryptoRecommendation(
            scratchpad="DEBUG dump only",
            reasoning="DEBUG mode: saved prompt and inputs to tmp/llm_dumps; no LLM call and no trades.",
            recommendations=[],
        )

    return invoke_llm(
        prompt,
        all_data,
        model=MultiCryptoRecommendation,
        with_fallback=with_fallback,
        **kwargs,
    )


def get_rebalance_recommendation(
    crypto_data_list: list[dict],
    indices_csv: str,
    balances: dict[str, dict],
    total_coin_value: int,
):
    """LLM을 사용하여 암호화폐 투자 추천을 받습니다."""
    # 각 코인별 데이터를 하나의 문자열로 조합
    data_descriptions = []
    for data in crypto_data_list:
        symbol = data["symbol"]
        description = f"""
=== {symbol} Data ===
User's current balance data in KRW in JSON
```json
{symbol}_balance_json
```
Recent trading data in KRW in JSON
```json
{symbol}_json_data
```
Historical data in USD in CSV
```csv
{symbol}_crypto_data_csv
```"""

        if data["network_stats_csv"]:  # BTC인 경우
            description += f"""
Network stats in CSV
```csv
{symbol}_network_stats_csv
```"""

        description += f"""
News in CSV
```csv
{symbol}_crypto_news_csv
```"""
        description = re.sub(
            rf"^({symbol}_(balance_json|json_data|crypto_data_csv|network_stats_csv|crypto_news_csv))",
            r"{\1}",
            description,
            flags=re.MULTILINE,
        )
        data_descriptions.append(description)

    all_data = "\n\n".join(data_descriptions)
    all_data += """
=== Market Indices ===
Indices data in USD in CSV
```csv
{indices_csv}
```"""

    # 각 코인별 데이터를 개별 변수로 전달하기 위한 kwargs 구성
    kwargs = {
        "indices_csv": indices_csv,
    }

    # 각 코인별로 데이터 변수 추가
    for data in crypto_data_list:
        symbol = data["symbol"]
        balance = balances.get(symbol, {})
        kwargs.update(
            {
                f"{symbol}_balance_json": json.dumps(balance),
                f"{symbol}_json_data": json.dumps(data["input_data"]),
                f"{symbol}_crypto_data_csv": data["crypto_data_csv"],
                f"{symbol}_crypto_news_csv": data["crypto_news_csv"],
            }
        )
        if data["network_stats_csv"]:  # BTC인 경우
            kwargs[f"{symbol}_network_stats_csv"] = data["network_stats_csv"]

    krw_balance = int(float(balances["KRW"]["quantity"] or 0))
    prompt = f"""You are a cryptocurrency portfolio rebalancing expert with exceptional risk management skills. You have access to:
 - Real-time market data, historical prices, volatility, news, and market sentiment
 - KRW balance: {krw_balance:,} KRW
 - Total coin value: {total_coin_value:,} KRW
 - Total portfolio value: {total_coin_value + krw_balance:,} KRW

Portfolio Value Calculation (CRITICAL - FOLLOW EXACTLY):
1. Calculate weights:
   - For each coin: weight = (current_value from current balance data / total portfolio value) × 100
   - KRW weight = (KRW balance / total portfolio value) × 100
   - Verify: Sum of ALL weights (including KRW) must equal 100%

2. Validation checks:
   - Each coin value must be < total portfolio value
   - Each weight must be < 100%
   - Sum of all weights must equal 100%
   - Total crypto weight = 100% - KRW weight

Rebalancing Rules:
1) Portfolio Composition
   - Suggest optimal weight for each cryptocurrency
   - Total crypto weight should be 70-90% (leaving 10-30% in KRW)
   - Propose rebalancing when current vs target value difference exceeds ±5%

2) Risk Management
   - Single coin maximum weight: 50% of total portfolio
   - Lower allocation for high-volatility coins
   - Higher allocation for top market cap coins

3) Trade Execution Criteria
   - Consider fees (0.04% round-trip) and only rebalance when deviation > 0.1%
   - Recommend splitting large orders into smaller ones
   - Use limit orders to minimize market impact

4) Market Context
   - Incorporate last 7 days of news and market sentiment
   - Analyze overall market trends and individual coin momentum
   - Consider correlation with major market indicators

Provide a clear and concise analysis in Korean (maximum 4000 characters). Format your response as follows:

1. 현재 포트폴리오 분석
- 총 포트폴리오 가치: XXX원
- KRW: XX.XX% (XXX원)
- 총 코인 가치: XX.XX% (XXX원)
- 코인별 상세:
  BTC: XX.XX% (X.XXX개 × 현재가 XXX원 = XXX원)
  ETH: XX.XX% (X.XXX개 × 현재가 XXX원 = XXX원)
  ...
- 리스크 평가
- 장단점

2. 시장 분석
- 주요 코인별 기술적/펀더멘털 분석
- 주요 뉴스 영향
- 시장 전망

3. 리밸런싱 제안
- 목표 비중 (전/후 각각 합계 100%가 되어야 함):
  BTC: XX.XX% -> XX.XX%
  ETH: XX.XX% -> XX.XX%
  ...
  KRW: XX.XX% -> XX.XX%
- 매매 계획 (우선순위 순):
  1) XXX: 매수/매도 (X.XXX개 × 현재가 XXX원 = XXX원)
  2) XXX: 매수/매도 (X.XXX개 × 현재가 XXX원 = XXX원)
  ...

4. 리스크 관리
- 손절매 기준
- 변동성 대비책
- 비상 상황 대응

Use simple text format without special characters. Focus on clear numerical values and specific recommendations. Double-check all calculations for accuracy.
    """
    if settings.DEBUG:
        _dump_llm_inputs(
            name="rebalance",
            prompt=prompt,
            human_message_template=all_data,
            template_kwargs=kwargs,
            user_slug="rebalance",
            template_format="f-string",
        )

        return "DEBUG dump only"

    return invoke_llm(prompt, all_data, with_anthropic=True, **kwargs)


def _decimal_to_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _round_price_to_unit(price: Decimal, unit: Decimal, rounding) -> Decimal:
    if unit <= 0:
        return price
    return (price / unit).to_integral_value(rounding=rounding) * unit


def _compute_market_order_limit_price(
    action: str,
    snapshot: dict,
    market: Optional[dict] = None,
) -> Optional[int | float]:
    """Add a light slippage guard to market orders only when spreads widen."""
    market = market or {}
    try:
        spread_pct = Decimal(str(snapshot.get("spread_pct")))
        best_ask = Decimal(str(snapshot.get("best_ask")))
        best_bid = Decimal(str(snapshot.get("best_bid")))
        price_unit = Decimal(str(market.get("price_unit")))
    except Exception:
        return None

    if (
        spread_pct < MARKET_LIMIT_PRICE_MIN_SPREAD_PCT
        or best_ask <= 0
        or best_bid <= 0
        or price_unit <= 0
    ):
        return None

    extra_pct = spread_pct / Decimal("2")
    extra_pct = max(
        MARKET_LIMIT_PRICE_MIN_EXTRA_PCT,
        min(MARKET_LIMIT_PRICE_MAX_EXTRA_PCT, extra_pct),
    )

    if action == "BUY":
        raw_price = best_ask * (Decimal("1") + (extra_pct / Decimal("100")))
        rounded_price = _round_price_to_unit(raw_price, price_unit, ROUND_UP)
        rounded_price = max(rounded_price, best_ask)
    elif action == "SELL":
        raw_price = best_bid * (Decimal("1") - (extra_pct / Decimal("100")))
        rounded_price = _round_price_to_unit(raw_price, price_unit, ROUND_DOWN)
        rounded_price = min(rounded_price, best_bid)
    else:
        return None

    min_price_raw = market.get("min_price")
    max_price_raw = market.get("max_price")
    if min_price_raw is not None:
        rounded_price = max(rounded_price, Decimal(str(min_price_raw)))
    if max_price_raw is not None:
        rounded_price = min(rounded_price, Decimal(str(max_price_raw)))

    return _decimal_to_number(rounded_price)


def send_trade_result(trading: Trading, balances: dict, chat_id: str):
    """거래 결과를 확인하고 텔레그램 메시지를 전송합니다."""
    symbol = trading.coin
    quantity = Decimal(trading.executed_qty or 0)
    amount = int(quantity * (trading.average_executed_price or 0))

    message_lines = [
        f"{trading.side}: {format_quantity(quantity)} {symbol} ({amount:,} 원)"
    ]
    if quantity:
        coin_quantity = Decimal(balances[symbol]["available"])
        coin_value = coin_quantity * trading.price
        krw_amount = Decimal(balances["KRW"]["available"])
        message_lines.append(
            f"보유: {format_quantity(coin_quantity)} {symbol} {coin_value:,.0f} / {krw_amount:,.0f} 원"
        )
        price_msg = "{:,.0f}".format(trading.average_executed_price or 0)
        message_lines.append(f"{symbol} 거래 가격: {price_msg} 원")

    if trading.reason:
        message_lines.append(trading.reason)

    if not quantity:
        order = (
            f"추천 매수금액: {trading.amount:,.0f} 원"
            if trading.side == "BUY"
            else f"추천 매도수량: {format_quantity(trading.quantity)} {symbol}"
        )
        message_lines.append(
            f"주문 취소됨! 주문하는게 좋다고 판단하면 직접 주문하세요. {trading.side} / {order}"
        )

    send_message("\n".join(message_lines), chat_id=chat_id)


def process_trade(
    user: User,
    symbol: str,
    order_detail: dict,
    chat_id: str,
    reason: str,
    crypto_price: float,
    amount: float = None,
    quantity: float = None,
    limit_price: float = None,
):
    """거래를 처리하고 결과를 저장 및 전송합니다."""
    order_data = order_detail["order"]
    trading = Trading.objects.create(
        user=user,
        order_id=order_data["order_id"],
        coin=symbol,
        amount=amount,
        quantity=quantity,
        limit_price=limit_price,
        reason=reason,
        price=crypto_price,
        type=order_data["type"],
        side=order_data["side"],
        status=order_data["status"],
        fee=order_data["fee"],
        order_detail=order_detail,
    )

    # current balance and value after order
    balances = coinone.get_balances()
    send_trade_result(balances=balances, chat_id=chat_id, trading=trading)

    return balances, trading


def execute_trade(
    user,
    recommendation: Recommendation,
    crypto_data: dict,
    chat_id: str,
    market: Optional[dict] = None,
) -> dict:
    """거래를 실행하고 결과를 처리합니다."""
    action = recommendation.action
    symbol = recommendation.symbol
    crypto_price = crypto_data["input_data"]["current_price"]
    snapshot = (crypto_data or {}).get("input_data") or {}
    effective_limit_price = recommendation.limit_price

    computed_limit_price = _compute_market_order_limit_price(
        action=action,
        snapshot=snapshot,
        market=market,
    )
    if computed_limit_price is not None:
        effective_limit_price = computed_limit_price

    logging.info(f"{recommendation=}")
    if effective_limit_price is not None:
        logging.info(
            "Applying Coinone market limit guard: symbol=%s action=%s limit_price=%s spread_pct=%s",
            symbol,
            action,
            effective_limit_price,
            snapshot.get("spread_pct"),
        )

    if settings.DEBUG:
        return

    if action == "BUY":
        amount = recommendation.amount
        if not amount:
            raise ValueError("amount is required for buy order")

        order = coinone.buy_ticker(symbol, amount, limit_price=effective_limit_price)
    elif action == "SELL":
        quantity = recommendation.quantity
        if not quantity:
            raise ValueError("quantity is required for sell order")

        order = coinone.sell_ticker(
            symbol,
            quantity,
            limit_price=effective_limit_price,
        )
    else:
        raise ValueError(f"Invalid action: {action}")

    logging.info(f"{action} order: {order}")

    if not order.order_id:
        raise ValueError(f"Failed to execute {action} order: {order}")

    order_detail = coinone.get_order_detail(order.order_id, symbol)
    logging.info(f"order_detail: {order_detail}")

    return process_trade(
        user,
        symbol=symbol,
        amount=recommendation.amount,
        quantity=recommendation.quantity,
        limit_price=effective_limit_price,
        crypto_price=crypto_price,
        order_detail=order_detail,
        chat_id=chat_id,
        reason=recommendation.reason,
    )


def auto_trading():
    """암호화폐 매매 프로세스를 실행합니다."""
    return _auto_trading()


def _auto_trading(force_run_outside_slots: bool = False):
    """암호화폐 매매 프로세스를 실행합니다.

    Args:
        force_run_outside_slots: True면 슬롯 외 시간대에도 실행합니다.
    """
    mt = compute_market_time(timezone.now())
    if not mt.should_run:
        if not force_run_outside_slots:
            logging.info(
                f"auto_trading: outside session-slot windows; {mt.market_time_context}"
            )
            return
        else:
            logging.info(
                f"_auto_trading: forcing run outside session-slot windows; {mt.market_time_context}"
            )

    market_time_context = mt.market_time_context

    # 오늘 날짜와 한 달 전 날짜 설정
    end_date = timezone.localdate()
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # 시장 지표 데이터 가져오기
    indices_data_csv = crypto.get_market_indices(start_date)

    # 전체 종목 정보 가져오기 (qty_unit 정보 포함)
    markets = coinone.get_markets()

    # 활성화된 트레이딩 설정에서 모든 target_coins를 가져와서 중복 제거
    active_configs = TradingConfig.objects.filter(is_active=True)
    if settings.DEBUG:
        # DEBUG: run dump-only for superuser only.
        active_configs = active_configs.filter(user__is_superuser=True)

    target_coins = set()
    for config in active_configs:
        target_coins.update(config.target_coins)

    # 모든 코인의 데이터 수집
    news_start_date = (end_date - timedelta(days=7)).strftime("%Y-%m-%d")
    crypto_data_dict = {}
    for symbol in target_coins:
        try:
            crypto_data = collect_crypto_data(symbol, news_start_date)
            crypto_data_dict[symbol] = crypto_data
        except Exception as e:
            logging.error(f"Failed to collect data for {symbol}: {e}")
            continue

    # 각 활성화된 유저별로 처리
    for config in active_configs:
        config: TradingConfig = config
        chat_id = config.telegram_chat_id

        # initialize coinone
        coinone.init(
            access_key=config.coinone_access_key,
            secret_key=config.coinone_secret_key,
        )

        balances = coinone.get_balances()

        # 해당 유저의 target_coins에 대한 데이터만 필터링하고 현재 잔고 가치 계산
        user_crypto_data = {}
        total_coin_value = 0
        for symbol in config.target_coins:
            if symbol in crypto_data_dict:
                data = dict(crypto_data_dict[symbol])
                balance = balances.get(symbol)
                if balance:
                    current_value = (
                        float(balance.get("available") or 0)
                        * data["input_data"]["current_price"]
                    )
                    data["input_data"]["current_value"] = current_value
                    total_coin_value += current_value
                user_crypto_data[symbol] = data

        current_prices = {
            symbol: data["input_data"]["current_price"]
            for symbol, data in user_crypto_data.items()
            if (data.get("input_data") or {}).get("current_price") is not None
        }
        recent_trades_csv = Trading.get_recent_trades_csv(
            user=config.user,
            current_prices=current_prices,
        )

        # LLM에게 추천 받기
        result, exc = [None] * 2
        # 최대 2번 시도
        for i in range(2):
            try:
                result = get_multi_recommendation(
                    list(user_crypto_data.values()),
                    indices_data_csv,
                    balances,
                    int(total_coin_value),
                    markets,
                    recent_trades_csv,
                    config,
                    market_time_context=market_time_context,
                    with_fallback=i > 0,
                )
                break
            except Exception as e:
                logging.warning(e)
                exc = e

        if not result and exc:
            logging.exception(
                f"Error getting multi recommendation for {config.user}: {exc}"
            )
            continue

        # DEBUG: dump-only mode (LLM not invoked; do not send telegram; do not trade)
        if settings.DEBUG:
            return

        # 분석 결과 전송
        send_message(
            f"```\n코인 분석:\n{result.scratchpad}\n\n{result.reasoning}```",
            chat_id=chat_id,
            is_markdown=True,
        )

        final_balances = None

        skipped = []

        max_trades = getattr(config, "max_trades_per_session", 2)
        recommendations_to_execute = result.recommendations[:max_trades]

        if len(result.recommendations) > max_trades:
            skipped.extend(
                [
                    {
                        "symbol": rec.symbol,
                        "action": rec.action,
                        "reason": f"exceeds max_trades_per_session ({max_trades})",
                    }
                    for rec in result.recommendations[max_trades:]
                ]
            )

        # 추천받은 거래 실행
        for recommendation in recommendations_to_execute:
            symbol = recommendation.symbol
            crypto_data = user_crypto_data[symbol]

            snapshot = (crypto_data or {}).get("input_data") or {}
            spread_pct = snapshot.get("spread_pct")
            best_bid = snapshot.get("best_bid")
            best_ask = snapshot.get("best_ask")
            action = recommendation.action

            # Hard safety filter for abnormal/illiquid orderbooks.
            should_skip = False
            skip_reason = None
            try:
                spread_val = float(spread_pct) if spread_pct is not None else None
            except Exception:
                spread_val = None

            if spread_val is None:
                should_skip = True
                skip_reason = "missing spread_pct"
            elif spread_val >= 2.0:
                should_skip = True
                skip_reason = f"spread_pct {spread_val:.2f}% >= 2.00%"
            elif action == "BUY" and spread_val >= 1.0:
                should_skip = True
                skip_reason = f"spread_pct {spread_val:.2f}% >= 1.00% (BUY blocked)"

            if should_skip:
                skipped.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "spread_pct": spread_pct,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "reason": skip_reason,
                    }
                )
                continue

            try:
                execute_trade(
                    config.user,
                    recommendation=recommendation,
                    crypto_data=crypto_data,
                    chat_id=chat_id,
                    market=markets.get(symbol, {}),
                )
            except Exception as e:
                logging.exception(f"Error executing trade for {symbol}: {e}")

        if skipped:
            lines = ["자동 안전필터로 일부 추천을 제외했습니다 (스프레드/유동성):"]
            for x in skipped:
                spread_txt = (
                    f"{float(x['spread_pct']):.2f}%"
                    if x.get("spread_pct") is not None
                    else "null"
                )
                bid_txt = x.get("best_bid")
                ask_txt = x.get("best_ask")
                lines.append(
                    f"- {x['action']} {x['symbol']}: spread={spread_txt}, bid={bid_txt}, ask={ask_txt} ({x['reason']})"
                )
            send_message("\n".join(lines), chat_id=chat_id, is_markdown=False)

        # 현재 잔고 가치 저장
        final_balances = coinone.get_balances()

        krw_balance = int(float(final_balances["KRW"]["available"]))
        total_coin_value = 0
        balances = []
        for symbol, balance in final_balances.items():
            crypto_data = user_crypto_data.get(symbol)
            if not crypto_data:
                continue

            available = Decimal(balance["available"])
            if available == 0 or float(balance["average_price"]) == 0:
                continue

            current_price = crypto_data["input_data"]["current_price"]
            krw_value = int(available * Decimal(current_price))
            if krw_value < 5000:
                continue

            balance.update(current_price=current_price, krw_value=krw_value)
            total_coin_value += krw_value
            balances.append(balance)

        balances = sorted(balances, key=lambda x: x["krw_value"], reverse=True)
        total_portfolio_value = krw_balance + total_coin_value

        Portfolio.objects.create(
            user=config.user,
            balances=balances,
            total_portfolio_value=total_portfolio_value,
            krw_balance=krw_balance,
            total_coin_value=total_coin_value,
        )


def rebalance_portfolio():
    # 오늘 날짜와 한 달 전 날짜 설정
    end_date = timezone.localdate()
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # 시장 지표 데이터 가져오기
    indices_data_csv = crypto.get_market_indices(start_date)

    # 현재 잔고 조회
    balances = upbit.get_available_balances()

    target_coins = set()
    for symbol in balances.keys():
        if symbol != "KRW":
            target_coins.add(symbol)

    # 모든 코인의 데이터 수집
    news_start_date = (end_date - timedelta(days=7)).strftime("%Y-%m-%d")
    crypto_data_dict = {}
    total_coin_value = 0
    for symbol in target_coins:
        try:
            crypto_data = collect_crypto_data(symbol, news_start_date, from_upbit=True)
            balance = balances.get(symbol)
            if balance:
                current_value = (
                    float(balance.get("quantity", 0))
                    * crypto_data["input_data"]["current_price"]
                )
                crypto_data["input_data"]["current_value"] = current_value
                total_coin_value += current_value
            crypto_data_dict[symbol] = crypto_data
        except Exception as e:
            logging.exception(f"Failed to collect data for {symbol}: {e}")
            continue

    config = TradingConfig.objects.filter(user__is_superuser=True).first()
    chat_id = config.telegram_chat_id

    # 해당 유저의 target_coins에 대한 데이터만 필터링
    crypto_data = {
        symbol: crypto_data_dict[symbol]
        for symbol in target_coins
        if symbol in crypto_data_dict
    }

    # LLM에게 추천 받기
    result, exc = [None] * 2
    # 최대 2번 시도
    for _ in range(2):
        try:
            result = get_rebalance_recommendation(
                list(crypto_data.values()),
                indices_data_csv,
                balances,
                int(total_coin_value),
            )
            break
        except Exception as e:
            logging.warning(e)
            exc = e

    if result:
        send_message(f"```\n{result}```", chat_id=chat_id, is_markdown=True)
    elif exc:
        logging.exception(f"Error getting rebalance recommendation: {exc}")


def fetch_crypto_listings():
    """CoinMarketCap에서 암호화폐 목록을 가져와 저장합니다."""
    min_market_cap = 1_000
    data = crypto.get_latest_listings(min_market_cap=min_market_cap)

    listing = []
    for coin in data:
        quote = coin["quote"]["USD"]
        market_cap = quote["market_cap"]

        if market_cap >= min_market_cap:
            listing.append(
                CryptoListing(
                    name=coin["name"],
                    symbol=coin["symbol"],
                    data_at=coin["last_updated"],
                    rank=coin["cmc_rank"],
                    circulating_supply=coin["circulating_supply"],
                    total_supply=coin["total_supply"],
                    max_supply=coin["max_supply"],
                    price=quote["price"],
                    market_cap=market_cap,
                    change_1h=quote["percent_change_1h"],
                    change_24h=quote["percent_change_24h"],
                    change_7d=quote["percent_change_7d"],
                    volume_24h=quote["volume_24h"],
                    raw=coin,
                )
            )

    result = CryptoListing.objects.bulk_create(listing)
    logging.info(f"fetch_crypto_listings: {len(result)}")


def select_coins_to_buy():
    """구매할 코인을 선택하고 결과를 알립니다."""
    today = timezone.now().date()
    start_date = today - timedelta(days=4)

    # 최근 5일 동안 24시간 변동률이 모두 0.5% 이상인 코인을 선택하고 필요한 정보를 한번에 가져옴
    coins = (
        CryptoListing.objects.filter(
            data_at__date__range=(start_date, today),
            market_cap__gt=10_000_000,
            volume_24h__gt=100_000,
            rank__lt=300,
        )
        .values("symbol")
        .annotate(
            count_positive=Count(Case(When(change_24h__gte=0.5, then=1))),
            first_price=Min("price"),
            last_price=Max("price"),
            avg_market_cap=Avg("market_cap"),
            name=F("name"),
        )
        .filter(count_positive=5)
        .annotate(
            change_5d=ExpressionWrapper(
                (F("last_price") - F("first_price")) / F("first_price") * 100,
                output_field=FloatField(),
            )
        )
        .order_by("-change_5d", "-avg_market_cap")[:10]
    )

    text_list = []

    # 선택된 코인 정보 출력
    for i, coin in enumerate(coins, 1):
        text_list.extend(
            [f"{i}. {coin['name']} ({coin['symbol']}) ${coin['last_price']:.4f}"]
        )
        text_list.append(f"Price 5 days ago: ${coin['first_price']:.4f}")
        text_list.append(f"Change over 5 days: {coin['change_5d']:.2f}%")

        market_cap = format_currency(coin["avg_market_cap"])
        text_list.extend([f"Average Market Cap: {market_cap}", ""])

    if text_list:
        text = "\n".join(text_list)
        text = f"Selected Coins to Buy:\n```\n{text}```"
    else:
        text = "No coins met the criteria for buying"

    config = TradingConfig.objects.filter(user__is_superuser=True).first()
    send_message(text, chat_id=config.telegram_chat_id, is_markdown=True)


def threads_post():
    prompt = """Analyze and summarize current market status in a Threads-friendly format:

US Stock Market 🇺🇸
- How are major indices performing? (S&P 500, NASDAQ, DOW)
- Which sectors are hot today?
- Any notable company movements?
- Key economic news?
- What's the overall market vibe?

Crypto Market 🌐
- How's Bitcoin doing?
- Which altcoins are making moves?
- Any big project news?
- Overall crypto market sentiment?

Requirements:
- Write in a casual, conversational tone
- Use line breaks between topics
- Start with an engaging hook
- Add relevant emojis (1-2 per point)
- Focus on what young investors care about
- Include only the most impactful numbers
- Keep each point short and snappy
- End with a key takeaway or tip"""

    system_instruction = [
        "You are a trendy financial content creator for Threads.",
        "Write in plain text format - NO markdown, NO bullet points, NO special formatting.",
        "Use emojis naturally, like in casual texting.",
        "Keep paragraphs short - 1-2 sentences max.",
        "Use line breaks to separate topics.",
        "Write in Korean with a casual, friendly tone.",
        "Avoid any special characters or formatting.",
        "Make it feel like a friend sharing market insights.",
        "Keep total length under 2000 characters.",
        "End with an engaging question or call to action.",
    ]

    result = invoke_gemini_search(prompt, system_instruction)
    print(result)
    return result


def buy_upbit_coins():
    now = timezone.localtime()

    # constance 설정 확인
    dca_enabled = constance.config.UPBIT_DCA_ENABLED
    auto_buy_enabled = constance.config.UPBIT_AUTO_BUY_ENABLED

    # 오전 6시에는 DCA 매수 (설정이 활성화된 경우)
    if now.hour == 6 and now.minute < 5 and dca_enabled:
        _buy_upbit_dca()
    # 다른 시간에는 자동 매수 (설정이 활성화된 경우)
    elif auto_buy_enabled:
        _buy_upbit_coins()

    update_upbit_portfolio()


def update_upbit_portfolio():
    data = upbit.get_balance_data()
    Portfolio.objects.create(
        exchange=ExchangeChoices.UPBIT,
        balances=data["balances"],
        total_portfolio_value=data["total_value"],
        krw_balance=data["krw_value"],
        total_coin_value=data["total_coin_value"],
    )


def get_coin_amounts(coins):
    # Constance 금액 맵 기준으로만 매수 금액 산출
    amount_map = _parse_amount_map()
    if not amount_map:
        return None

    return {coin: amount_map[coin] for coin in coins if coin in amount_map}


def _parse_ath_map():
    ath_map_raw = constance.config.UPBIT_ATH_MAP_KRW
    if not ath_map_raw:
        return None

    try:
        ath_map = ath_map_raw
        if not isinstance(ath_map, dict) or not ath_map:
            return None
        return ath_map
    except (TypeError, ValueError) as e:
        logging.warning(f"Invalid ATH map: {ath_map_raw}, error: {e}")
        return None


def _parse_amount_map():
    # 코인별 매수 금액 맵 (JSON/Dict) 파싱
    amount_map_raw = constance.config.UPBIT_AMOUNT_MAP_KRW
    if not amount_map_raw:
        return None

    try:
        amount_map = amount_map_raw
        if not isinstance(amount_map, dict) or not amount_map:
            return None
        return amount_map
    except (TypeError, ValueError) as e:
        logging.warning(f"Invalid amount map: {amount_map_raw}, error: {e}")
        return None


def _filter_coins_by_ath(coins):
    ath_map = _parse_ath_map()
    if not ath_map:
        return None

    filtered = {coin for coin in coins if coin in ath_map}
    return filtered


def _get_tiered_threshold(drawdown_pct: float) -> float:
    """Get tiered price-drop threshold based on ATH drawdown percentage.

    Args:
        drawdown_pct: Current drawdown from ATH as percentage (0-100+)

    Returns:
        Required price-drop percentage threshold

    Tiers:
    - 0-20% drawdown: 1.0% threshold
    - 20-40% drawdown: 2.0% threshold
    - 40-60% drawdown: 3.0% threshold
    - 60-80% drawdown: 4.0% threshold
    - >80% drawdown: 5.0% threshold (capped)
    """
    if drawdown_pct < 0:
        return 1.0
    elif drawdown_pct < 20:
        return 1.0
    elif drawdown_pct < 40:
        return 2.0
    elif drawdown_pct < 60:
        return 3.0
    elif drawdown_pct < 80:
        return 4.0
    else:
        return 5.0


def _check_extra_drop_exception(coin, last_price, last_trading, extra_drop_threshold):
    if not last_trading:
        return False

    ath_map = _parse_ath_map()
    if not ath_map or coin not in ath_map:
        return False

    ath_price = Decimal(str(ath_map[coin]))

    last_buy_price = last_trading.average_price
    last_buy_at = last_trading.created

    if not last_buy_price:
        return False

    current_drawdown_pct = (ath_price - Decimal(str(last_price))) / ath_price * 100
    last_buy_drawdown_pct = (ath_price - last_buy_price) / ath_price * 100

    additional_drop_pct = current_drawdown_pct - last_buy_drawdown_pct

    return additional_drop_pct >= extra_drop_threshold


def _buy_upbit_coins():
    data = upbit.get_balance_data()
    balances, total_value, krw_value = dict_at(
        data, "balances", "total_value", "krw_value"
    )

    # 총 자산이 4억 원 이상이면 구매 중지
    if total_value >= 400_000_000:
        return

    coins = {balance["symbol"].split(".")[0] for balance in balances}

    # ATH 필터 적용
    filtered_coins = _filter_coins_by_ath(coins)
    if filtered_coins is None:
        logging.warning("ATH map is empty or invalid, skipping auto-buy")
        return

    if not filtered_coins:
        logging.info("No coins in ATH map, skipping auto-buy")
        return

    coins = filtered_coins
    coin_amounts = get_coin_amounts(coins)

    if coin_amounts is None:
        logging.warning("Amount map is empty or invalid, skipping auto-buy")
        return

    if not coin_amounts:
        logging.info("No coins in amount map, skipping auto-buy")
        return

    missing_coins = set(coins) - set(coin_amounts.keys())
    if missing_coins:
        logging.warning(f"Coins missing from amount map (skipped): {missing_coins}")

    # 원화 잔고가 코인 구매에 필요한 금액보다 적으면 구매 중지
    required_krw = sum(coin_amounts.values())
    logging.info(f"{required_krw=:,} {krw_value=:,.0f}")
    if krw_value < required_krw:
        return

    today = timezone.localdate()
    extra_drop_threshold = constance.config.UPBIT_ATH_EXTRA_DROP_PCT
    ath_map = _parse_ath_map()

    # 코인 구매
    for coin, amount in coin_amounts.items():
        last_trading = next(
            (
                UpbitTrading.objects.filter(coin=coin, is_dca=is_dca)
                .order_by("-created")
                .first()
                for is_dca in (False, True)
            ),
            None,
        )

        if not last_trading:
            # last_trading이 없으면 balances에서 수익률 확인
            coin_balance = next((b for b in balances if b.get("symbol") == coin), None)
            if not coin_balance or not coin_balance.get("avg_buy_price"):
                continue

            avg_buy_price = Decimal(str(coin_balance["avg_buy_price"]))
            current_price = Decimal(str(coin_balance["current_price"]))

            if ath_map and coin in ath_map:
                ath_price = Decimal(str(ath_map[coin]))
                drawdown_pct = float((ath_price - current_price) / ath_price * 100)
                threshold = _get_tiered_threshold(drawdown_pct)
            else:
                drawdown_pct = None
                threshold = 1.0

            price_change = (current_price - avg_buy_price) / avg_buy_price * 100
            should_buy = price_change <= -threshold
            logging.info(
                f"{coin}: {should_buy=} {format_quantity(current_price)} <- {format_quantity(avg_buy_price)} ({price_change:.2f}%) "
                f"[no trading history, drawdown={drawdown_pct:.2f}%, threshold={threshold:.1f}%]"
            )
        else:
            last_buy_price = last_trading.average_price
            last_buy_at = last_trading.created

            last_candle = upbit.get_candles(coin, count=1)[0]
            last_price = Decimal(last_candle["trade_price"])

            if ath_map and coin in ath_map:
                ath_price = Decimal(str(ath_map[coin]))
                drawdown_pct = float((ath_price - last_price) / ath_price * 100)
                threshold = _get_tiered_threshold(drawdown_pct)
            else:
                drawdown_pct = None
                threshold = 1.0

            price_change = (last_price - last_buy_price) / last_buy_price * 100
            should_buy = (
                price_change <= -threshold
                and last_buy_at < timezone.now() - timedelta(hours=2)
            )

            # 일일 매수 제한: 오늘 이미 매수했는지 확인 (추가 하락 예외 적용)
            last_auto_buy_today = UpbitTrading.objects.filter(
                coin=coin, is_dca=False, created__date=today
            ).first()

            if last_auto_buy_today:
                # 추가 하락 예외 확인
                extra_drop_allowed = _check_extra_drop_exception(
                    coin, last_price, last_auto_buy_today, extra_drop_threshold
                )
                if not extra_drop_allowed:
                    logging.info(
                        f"{coin}: skipped (daily limit reached, no extra drop exception)"
                    )
                    should_buy = False
                else:
                    logging.info(
                        f"{coin}: extra drop exception triggered (additional drop >= {extra_drop_threshold}%)"
                    )

            logging.info(
                f"{coin}: {should_buy=} {format_quantity(last_price)} <- {format_quantity(last_buy_price)} ({price_change:.2f}%) "
                f"{last_buy_at} [drawdown={drawdown_pct:.2f}%, threshold={threshold:.1f}%]"
            )

        if should_buy:
            res = upbit.buy_coin(coin, amount)
            logging.info(f"{coin=} {amount=:,} {res=}")

            uuid = res["uuid"]

            # trades_count가 0이면 주문 체결 안된 것으로 판단
            for _ in range(10):
                detail = upbit.get_order_detail(uuid)
                if detail["trades_count"] > 0:
                    break

                time.sleep(0.1)

            UpbitTrading.objects.create(
                coin=coin,
                amount=amount,
                uuid=uuid,
                state=detail["state"],
                order_detail=detail,
            )

        time.sleep(0.1)


def _buy_upbit_dca():
    data = upbit.get_balance_data()
    balances, krw_value = dict_at(data, "balances", "krw_value")

    coins = {balance["symbol"].split(".")[0] for balance in balances}

    # ATH 필터 적용
    filtered_coins = _filter_coins_by_ath(coins)
    if filtered_coins is None:
        logging.warning("ATH map is empty or invalid, skipping DCA")
        return

    if not filtered_coins:
        logging.info("No coins in ATH map, skipping DCA")
        return

    coins = filtered_coins
    coin_amounts = get_coin_amounts(coins)

    if coin_amounts is None:
        logging.warning("Amount map is empty or invalid, skipping DCA")
        return

    if not coin_amounts:
        logging.info("No coins in amount map, skipping DCA")
        return

    missing_coins = set(coins) - set(coin_amounts.keys())
    if missing_coins:
        logging.warning(f"Coins missing from amount map (skipped): {missing_coins}")

    # 원화 잔고가 코인 구매에 필요한 금액보다 적으면 구매 중지
    required_krw = sum(coin_amounts.values())
    logging.info(f"{required_krw=:,} {krw_value=:,.0f}")
    if krw_value < required_krw:
        return

    today = timezone.localdate()

    # 코인 구매
    for coin, amount in coin_amounts.items():
        # 오늘 이미 매수한 코인이면 매수 안함
        if UpbitTrading.objects.filter(
            coin=coin, is_dca=True, created__date=today
        ).exists():
            continue

        res = upbit.buy_coin(coin, amount)
        logging.info(f"DCA: {coin=} {amount=:,} {res=}")

        uuid = res["uuid"]

        # trades_count가 0이면 주문 체결 안된 것으로 판단
        for _ in range(10):
            detail = upbit.get_order_detail(uuid)
            if detail["trades_count"] > 0:
                break

            time.sleep(0.1)

        UpbitTrading.objects.create(
            is_dca=True,
            coin=coin,
            amount=amount,
            uuid=uuid,
            state=detail["state"],
            order_detail=detail,
        )

        time.sleep(0.1)


class AutoTradingRunner:
    def __init__(self, config):
        self.config: TradingConfig = config
        self.user: User = config.user
        self.auto_trading: AutoTrading = None
        self.trading_obj: Trading = None

    def already_processing(self):
        latest = AutoTrading.objects.filter(is_processing=True).order_by("-id").first()
        return latest and not latest.is_expired

    def start(self):
        self.auto_trading = AutoTrading.objects.create()

    def finish(self):
        if self.auto_trading:
            self.auto_trading.is_processing = False
            self.auto_trading.finished_at = timezone.now()
            self.auto_trading.save()

    def run(self):
        if self.already_processing():
            logging.warning("Auto trading already in progress, skipping this run.")
            return
        self.start()
        try:
            self._main_logic()
        finally:
            self.finish()

    def _main_logic(self):
        coinone.init(
            access_key=self.config.coinone_access_key,
            secret_key=self.config.coinone_secret_key,
        )
        markets = coinone.get_markets()
        btc_market = markets.get("BTC", {})
        ticker = coinone.get_ticker("BTC")
        current_price = float(ticker.get("trade_price") or ticker.get("last") or 0)
        candle_data = coinone.get_candles("BTC", "15m", size=100)
        candles = candle_data.get("chart", [])
        prices = [float(c["close"]) for c in candles if "close" in c]
        orderbook = coinone.get_orderbook("BTC")
        balances = coinone.get_balances()
        krw_available = float(balances.get("KRW", {}).get("available") or 0)
        btc_data = balances.get("BTC", {})
        btc_available = float(btc_data.get("available") or 0)
        btc_avg_price = float(btc_data.get("average_price") or 0)
        algo_param = (
            AlgorithmParameter.objects.filter(user=self.user).order_by("-id").first()
        )
        if not algo_param:
            algo_param = AlgorithmParameter.objects.create(user=self.user)
        if len(prices) < max(algo_param.rsi_period, algo_param.bollinger_period):
            return
        prices.reverse()
        rsi_values = ta.calculate_rsi(prices, period=algo_param.rsi_period)
        latest_rsi = rsi_values[-1]
        middle, upper, lower = ta.calculate_bollinger_bands(
            prices,
            period=algo_param.bollinger_period,
            num_std=float(algo_param.bollinger_std),
        )
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        bid_prices = [float(bid["price"]) for bid in bids]
        ask_prices = [float(ask["price"]) for ask in asks]
        bid_volumes = [float(bid["qty"]) for bid in bids]
        ask_volumes = [float(ask["qty"]) for ask in asks]
        total_bid_volume = sum(bid_volumes[:5])
        total_ask_volume = sum(ask_volumes[:5])
        buy_pressure = (
            total_bid_volume / (total_bid_volume + total_ask_volume)
            if total_bid_volume + total_ask_volume > 0
            else 0.5
        )
        signal = ta.generate_trade_signal(
            buy_rsi_threshold=algo_param.buy_rsi_threshold,
            sell_rsi_threshold=algo_param.sell_rsi_threshold,
            current_price=current_price,
            rsi=latest_rsi,
            upper_band=upper,
            lower_band=lower,
        )
        btc_invested = btc_available * btc_avg_price
        btc_now = btc_available * current_price
        if btc_invested > 0:
            profit_rate = (btc_now - btc_invested) / btc_invested * 100
        else:
            profit_rate = 0

        # AutoTrading 지표/상태 저장
        self.auto_trading.rsi = latest_rsi
        self.auto_trading.bollinger_upper = upper
        self.auto_trading.bollinger_lower = lower
        self.auto_trading.signal = signal
        self.auto_trading.buy_pressure = buy_pressure
        self.auto_trading.current_price = current_price
        self.auto_trading.btc_available = btc_available
        self.auto_trading.btc_avg_price = btc_avg_price
        self.auto_trading.btc_profit_rate = profit_rate
        self.auto_trading.krw_available = krw_available
        self.auto_trading.save()

        # --- STOP LOSS/TAKE PROFIT 분할매도, 쿨타임 적용 ---
        stop_loss_signal = None
        now = timezone.now()
        if btc_available > 0:
            stop_loss_signal = ta.check_stop_loss_take_profit(
                btc_avg_price,
                current_price,
                stop_loss_pct=algo_param.stop_loss_pct,
                take_profit_pct=algo_param.take_profit_pct,
            )
            self.auto_trading.stop_loss_signal = stop_loss_signal
            self.auto_trading.save()

        logging.info(
            f"Signal={signal}/{stop_loss_signal} RSI={latest_rsi:.2f} Pressure={buy_pressure:.3f} Band={upper:,.0f}/{lower:,.0f}"
        )
        logging.info(
            f"BTC={btc_available} Profit={profit_rate:.2f}% Price={btc_avg_price:,.0f}/{current_price:,.0f} KRW={krw_available:,.0f}"
        )

        # STOP LOSS/TAKE PROFIT 분할매도
        if btc_available > 0 and stop_loss_signal in ["STOP LOSS", "TAKE PROFIT"]:
            sell_ratio = algo_param.sell_chunk_ratio
            sell_quantity = btc_available * sell_ratio
            # 수량이 0.00000001 미만이면 매도 안함
            if sell_quantity < 1e-8:
                return
            order = coinone.sell_ticker("BTC", sell_quantity, limit_price=bid_prices[0])
            if order and order.order_id:
                order_detail = coinone.get_order_detail(order.order_id, "BTC")
                balances, trading_obj = process_trade(
                    self.user,
                    symbol="BTC",
                    quantity=sell_quantity,
                    limit_price=bid_prices[0],
                    crypto_price=current_price,
                    order_detail=order_detail,
                    chat_id=self.config.telegram_chat_id,
                    reason=f"{stop_loss_signal} 조건 만족으로 분할 매도 실행",
                )
                self.auto_trading.trading = trading_obj
                self.auto_trading.save()
            save_portfolio_snapshot(self.user, balances)
            return

        # STOP LOSS 쿨타임 체크 (DB에서 최근 STOP LOSS 매도 시각 조회)
        cooldown = getattr(algo_param, "stop_loss_cooldown_minutes", 60)
        last_stop_loss = (
            AutoTrading.objects.filter(
                trading__side="SELL",
                trading__coin="BTC",
                stop_loss_signal="STOP LOSS",
            )
            .order_by("-created")
            .first()
        )
        if last_stop_loss:
            elapsed = (now - last_stop_loss.created).total_seconds() / 60
            if elapsed < cooldown:
                logging.info(
                    f"STOP LOSS 쿨타임 적용중: {elapsed:.1f}분 경과, {cooldown}분 대기 필요"
                )
                return

        # --- 분할매수/추가매수 로직 (최근 매도 이후 매수 기준) ---
        buy_ratio = getattr(algo_param, "buy_chunk_ratio", 0.5)
        min_trade_amount = float(btc_market.get("min_order_amount", 0))
        buy_amount = max(
            min_trade_amount, krw_available * algo_param.max_krw_buy_ratio * buy_ratio
        )
        max_additional_buys = getattr(algo_param, "max_additional_buys", 2)
        # 1) 가장 최근 SELL(매도) AutoTrading의 id 구하기
        last_sell = (
            AutoTrading.objects.filter(trading__side="SELL", trading__coin="BTC")
            .order_by("-created")
            .first()
        )
        if last_sell:
            # 2) 그 id 이후의 BUY(매수) AutoTrading 중 첫 번째는 최초매수, 그 이후만 추가매수로 카운트
            buys_after_sell = list(
                AutoTrading.objects.filter(
                    trading__side="BUY", trading__coin="BTC", id__gt=last_sell.id
                ).order_by("created")
            )
        else:
            buys_after_sell = list(
                AutoTrading.objects.filter(
                    trading__side="BUY", trading__coin="BTC"
                ).order_by("created")
            )
        # 최초매수 이후의 추가매수 카운트
        additional_buy_count = max(0, len(buys_after_sell) - 1)
        add_buy_allowed = (
            btc_available > 0
            and additional_buy_count < max_additional_buys
            and latest_rsi <= getattr(algo_param, "add_buy_rsi_threshold", 20.0)
            and (
                current_price
                <= lower
                + getattr(algo_param, "add_buy_bollinger_band", -1.5)
                * (upper - middle)
                / 2
            )
        )
        # 신규매수 or 추가매수
        if signal == "BUY" and krw_available >= self.config.min_trade_amount:
            if (
                (latest_rsi < algo_param.buy_rsi_threshold or current_price <= lower)
                and buy_pressure >= algo_param.buy_pressure_threshold
                and (
                    profit_rate <= algo_param.buy_profit_rate
                    or btc_available == 0
                    or add_buy_allowed
                )
            ):
                order = coinone.buy_ticker("BTC", buy_amount)
                if order and order.order_id:
                    order_detail = coinone.get_order_detail(order.order_id, "BTC")
                    balances, trading_obj = process_trade(
                        self.user,
                        symbol="BTC",
                        amount=buy_amount,
                        crypto_price=current_price,
                        order_detail=order_detail,
                        chat_id=self.config.telegram_chat_id,
                        reason="분할 매수/추가매수 실행",
                    )
                    self.auto_trading.trading = trading_obj
                    self.auto_trading.save()
                save_portfolio_snapshot(self.user, balances)
        # 분할매도(일반 매도 신호)
        elif signal == "SELL" and btc_available > 0:
            if (
                (latest_rsi > algo_param.sell_rsi_threshold or current_price >= upper)
                and buy_pressure < algo_param.sell_pressure_threshold
                and profit_rate >= algo_param.sell_profit_rate
            ):
                sell_ratio = getattr(algo_param, "sell_chunk_ratio", 0.5)
                sell_quantity = btc_available * sell_ratio
                if sell_quantity < 1e-8:
                    return
                order = coinone.sell_ticker(
                    "BTC", sell_quantity, limit_price=bid_prices[0]
                )
                if order and order.order_id:
                    order_detail = coinone.get_order_detail(order.order_id, "BTC")
                    balances, trading_obj = process_trade(
                        self.user,
                        symbol="BTC",
                        quantity=sell_quantity,
                        limit_price=bid_prices[0],
                        crypto_price=current_price,
                        order_detail=order_detail,
                        chat_id=self.config.telegram_chat_id,
                        reason="분할 매도 실행",
                    )
                    self.auto_trading.trading = trading_obj
                    self.auto_trading.save()
                save_portfolio_snapshot(self.user, balances)


def auto_trade_btc():
    config = TradingConfig.objects.get(user__is_superuser=True)
    runner = AutoTradingRunner(config)
    runner.run()


def optimize_parameters():
    # 일 1회 파라미터 최적화
    now = timezone.localtime()
    if AlgorithmParameter.objects.filter(created__date=now.date()).exists():
        return

    config = TradingConfig.objects.get(user__is_superuser=True)
    user = config.user
    # 최근 거래 데이터
    recent_trades = (
        Trading.objects.filter(user=user, coin="BTC", auto_tradings__isnull=False)
        .order_by("-id")[:50]
        .values()
    )
    recent_trades = [dict_omit(trade, "status") for trade in recent_trades]
    # 현재 시점의 포트폴리오를 생성하여 export
    coinone.init(
        access_key=config.coinone_access_key, secret_key=config.coinone_secret_key
    )
    balances = coinone.get_balances()
    balances = {
        k: v
        for k, v in balances.items()
        if k == "KRW"
        or (
            float(v.get("available") or 0) > 0
            and float(v.get("average_price") or 0) > 0
        )
    }
    btc_data = balances.get("BTC", {})
    btc_available = float(btc_data.get("available") or 0)
    ticker = coinone.get_ticker("BTC")
    btc_price = float(ticker.get("trade_price") or ticker.get("last") or 0)
    krw_balance = int(float(balances["KRW"]["available"]))
    btc_value = int(btc_available * btc_price)
    total_portfolio_value = krw_balance + btc_value
    portfolio = Portfolio.objects.create(
        user=user,
        balances=balances,
        total_portfolio_value=total_portfolio_value,
        krw_balance=krw_balance,
        total_coin_value=btc_value,
    )
    portfolio_data = portfolio.export()
    payload = create_optimization_payload(
        trade_data=recent_trades,
        portfolio=portfolio_data,
    )
    optimized_params = request_optimized_parameters(payload)
    logging.info(f"{optimized_params=}")
    if optimized_params:
        AlgorithmParameter.objects.create(
            user=user,
            **optimized_params.model_dump(),
        )


def save_portfolio_snapshot(user, balances=None):
    balances = balances or coinone.get_balances()
    btc_price = float(coinone.get_ticker("BTC").get("trade_price") or 0)
    btc_available = float(balances.get("BTC", {}).get("available") or 0)
    krw_balance = int(float(balances["KRW"]["available"]))
    btc_value = int(btc_available * btc_price)
    total_portfolio_value = krw_balance + btc_value
    Portfolio.objects.create(
        user=user,
        balances=balances,
        total_portfolio_value=total_portfolio_value,
        krw_balance=krw_balance,
        total_coin_value=btc_value,
    )


def run_bybit_mechanical_trading(symbol="BTCUSDT", paper_mode=None):
    if paper_mode is None:
        paper_mode = getattr(settings, "BYBIT_PAPER_MODE", True)
    trader = BybitMechanicalTrader(symbol=symbol, paper_mode=paper_mode)
    trader.run()


def bybit_daily_review(notify=True):
    reviewer = BybitDailyReviewer()
    result = reviewer.run()
    if notify and result:
        _send_bybit_review_notification(result)


def _send_bybit_review_notification(result):
    try:
        config = TradingConfig.objects.filter(bybit_alert_enabled=True).first()
        if not config:
            return

        lines = [
            "📊 Bybit 일일 리뷰 완료",
            "",
            f"조회 기간: {result['lookback_hours']}시간",
            f"총 거래: {result['total_trades']}건",
            f"승률: {result['win_rate']:.1f}%",
            f"Profit Factor: {result['profit_factor']:.2f}",
            f"총 손익: ${result['total_pnl']:.2f}",
            "",
            f"파라미터 조정 필요: {'예' if result['should_adjust'] else '아니오'}",
        ]

        if result.get("reasoning"):
            lines.extend(["", f"분석: {result['reasoning']}"])

        send_message("\n".join(lines), chat_id=config.telegram_chat_id)
    except Exception as e:
        logging.error(f"Failed to send notification: {e}")


class BybitMechanicalTrader:
    def __init__(self, symbol="BTCUSDT", paper_mode=True):
        self.symbol = symbol
        self.paper_mode = paper_mode
        self.params = self._load_parameters()

    def _load_parameters(self):
        params = BybitMechanicalParameter.objects.first()
        if not params:
            params = BybitMechanicalParameter.objects.create()
            logging.info(f"Created default BybitMechanicalParameter (id={params.id})")
        return params

    def run(self):
        try:
            signal = self._generate_signal()

            if signal.action and self._should_enter(signal):
                self._execute_entry(signal)

            self._check_exits()
            self._log_status()

        except Exception as e:
            logging.exception(f"Error in mechanical trading cycle: {e}")

    def _calculate_adx(self, high, low, close, period=14):
        try:
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()

            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)

            atr = tr.rolling(window=period).mean()
            plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
            minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
            dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
            adx = dx.rolling(window=period).mean()
            return adx
        except Exception:
            return None

    def _compute_indicators(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rsi_series = calc_rsi(close, period=14)
        macd_line, signal_line, hist = calc_macd(close)
        hist_prev = hist.shift(1)
        ema20 = calc_ema(close, span=20)
        ema50 = calc_ema(close, span=50)
        vma20 = calc_volume_ma(volume, period=20)
        atr14 = calc_atr(high, low, close, period=14)
        adx = self._calculate_adx(high, low, close)
        bb_upper, bb_mid, bb_lower = calc_bb(
            close, period=self.params.bbands_period, num_std=self.params.bbands_std
        )
        atr_avg_series = atr14.rolling(window=self.params.atr_avg_period).mean()
        atr_current = float(atr14.iloc[-1])
        atr_avg = (
            float(atr_avg_series.iloc[-1]) if len(atr_avg_series) > 0 else atr_current
        )
        atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0

        last = df.iloc[-1]
        close_price = float(last["close"])
        bb_u = float(bb_upper.iloc[-1])
        bb_l = float(bb_lower.iloc[-1])
        bb_m = float(bb_mid.iloc[-1])
        bb_range = bb_u - bb_l
        bb_position = ((close_price - bb_l) / bb_range * 100) if bb_range > 0 else 50.0
        vol_ma = float(vma20.iloc[-1])
        volume_ratio = float(last["volume"]) / vol_ma if vol_ma > 0 else 1.0

        return {
            "close": close_price,
            "rsi": float(rsi_series.iloc[-1]),
            "macd": float(macd_line.iloc[-1]),
            "macd_signal": float(signal_line.iloc[-1]),
            "macd_hist": float(hist.iloc[-1]),
            "macd_hist_prev": float(hist_prev.iloc[-1]),
            "ema20": float(ema20.iloc[-1]),
            "ema50": float(ema50.iloc[-1]),
            "volume": float(last["volume"]),
            "volume_ma20": vol_ma,
            "volume_ratio": volume_ratio,
            "atr": atr_current,
            "atr_avg": atr_avg,
            "atr_ratio": atr_ratio,
            "adx": float(adx.iloc[-1]) if adx is not None else 25.0,
            "bb_upper": bb_u,
            "bb_mid": bb_m,
            "bb_lower": bb_l,
            "bb_position": bb_position,
        }

    def _check_entry_conditions(self, ind):
        p = self.params
        conditions_long = []
        conditions_short = []

        c1_volatility_breakout_long = ind["atr_ratio"] >= p.atr_breakout_ratio
        c1_volatility_breakout_short = ind["atr_ratio"] >= p.atr_breakout_ratio
        conditions_long.append(
            (
                "Volatility breakout (ATR {:.1f}x >= {:.1f}x)".format(
                    ind["atr_ratio"], p.atr_breakout_ratio
                ),
                c1_volatility_breakout_long,
            )
        )
        conditions_short.append(
            (
                "Volatility breakout (ATR {:.1f}x >= {:.1f}x)".format(
                    ind["atr_ratio"], p.atr_breakout_ratio
                ),
                c1_volatility_breakout_short,
            )
        )

        c2_rsi_long = ind["rsi"] < p.rsi_buy_threshold
        c2_rsi_short = ind["rsi"] > p.rsi_sell_threshold
        conditions_long.append(
            (
                "RSI oversold ({:.1f} < {:.1f})".format(
                    ind["rsi"], p.rsi_buy_threshold
                ),
                c2_rsi_long,
            )
        )
        conditions_short.append(
            (
                "RSI overbought ({:.1f} > {:.1f})".format(
                    ind["rsi"], p.rsi_sell_threshold
                ),
                c2_rsi_short,
            )
        )

        c2_bb_long = ind["bb_position"] < 40
        c2_bb_short = ind["bb_position"] > 60
        conditions_long.append(
            (
                "BB position near lower ({:.0f}% < 40%)".format(ind["bb_position"]),
                c2_bb_long,
            )
        )
        conditions_short.append(
            (
                "BB position near upper ({:.0f}% > 60%)".format(ind["bb_position"]),
                c2_bb_short,
            )
        )

        c3_trend_long = ind["close"] > ind["ema20"] and ind["ema20"] > ind["ema50"]
        c3_trend_short = ind["close"] < ind["ema20"] and ind["ema20"] < ind["ema50"]
        conditions_long.append(("EMA trend up (price > EMA20 > EMA50)", c3_trend_long))
        conditions_short.append(
            ("EMA trend down (price < EMA20 < EMA50)", c3_trend_short)
        )

        c4_momentum_long = (
            ind["macd_hist"] > 0 and ind["macd_hist"] > ind["macd_hist_prev"]
        )
        c4_momentum_short = (
            ind["macd_hist"] < 0 and ind["macd_hist"] < ind["macd_hist_prev"]
        )
        conditions_long.append(
            (
                "MACD momentum up (hist={:.2f} > 0, rising)".format(ind["macd_hist"]),
                c4_momentum_long,
            )
        )
        conditions_short.append(
            (
                "MACD momentum down (hist={:.2f} < 0, falling)".format(
                    ind["macd_hist"]
                ),
                c4_momentum_short,
            )
        )

        c5_volume_long = ind["volume_ratio"] >= p.volume_spike_ratio
        c5_volume_short = ind["volume_ratio"] >= p.volume_spike_ratio
        conditions_long.append(
            (
                "Volume spike ({:.1f}x >= {:.1f}x)".format(
                    ind["volume_ratio"], p.volume_spike_ratio
                ),
                c5_volume_long,
            )
        )
        conditions_short.append(
            (
                "Volume spike ({:.1f}x >= {:.1f}x)".format(
                    ind["volume_ratio"], p.volume_spike_ratio
                ),
                c5_volume_short,
            )
        )

        met_long = sum(1 for _, met in conditions_long if met)
        met_short = sum(1 for _, met in conditions_short if met)

        for label, met in conditions_long:
            if met:
                logging.info(f"  [LONG condition met] {label}")
        for label, met in conditions_short:
            if met:
                logging.info(f"  [SHORT condition met] {label}")

        return {
            "met_long": met_long,
            "met_short": met_short,
            "conditions_long": conditions_long,
            "conditions_short": conditions_short,
        }

    def _generate_signal(self):
        rows = bybit.get_kline(self.symbol, interval="5", limit=200, category="linear")
        df = bybit.klines_to_dataframe(rows)
        df = bybit.drop_unclosed_candle(df, interval_minutes=5)

        if len(df) < 50:
            logging.warning(f"Insufficient data: {len(df)} candles")
            return BybitSignalData(
                action=None, score_long=0, score_short=0, indicators={}
            )

        indicators = self._compute_indicators(df)
        cond = self._check_entry_conditions(indicators)
        p = self.params

        action = None
        score_gap = abs(cond["met_long"] - cond["met_short"])

        if cond["met_long"] >= 4 and score_gap >= 3:
            if cond["met_long"] > cond["met_short"]:
                action = "LONG"
        elif cond["met_short"] >= 4 and score_gap >= 3:
            if cond["met_short"] > cond["met_long"]:
                action = "SHORT"

        met_count = max(cond["met_long"], cond["met_short"]) if action else 0
        logging.info(
            f"Signal: {action or '-'} | Conditions met: LONG={cond['met_long']}/5 SHORT={cond['met_short']}/5 "
            f"| Score gap:{score_gap:.0f}/{p.min_score_gap:.0f}"
            f"| RSI:{indicators['rsi']:.1f} MACD:{indicators['macd_hist']:.4f} ADX:{indicators['adx']:.1f} "
            f"ATR:{indicators['atr_ratio']:.2f}x BB:{indicators['bb_position']:.0f}% Vol:{indicators['volume_ratio']:.1f}x"
        )

        self._save_signal(indicators, cond["met_long"], cond["met_short"], action)

        return BybitSignalData(
            action=action,
            score_long=cond["met_long"],
            score_short=cond["met_short"],
            indicators=indicators,
        )

    def _should_enter(self, signal):
        if not signal.action:
            return False

        open_trades = BybitMechanicalTrade.objects.filter(
            symbol=self.symbol, is_open=True, side=signal.action
        )

        return not open_trades.exists()

    def _execute_entry(self, signal):
        leverage = self._calculate_leverage(signal.indicators)
        position_size = self._calculate_position_size(leverage)

        trade = BybitMechanicalTrade.objects.create(
            symbol=self.symbol,
            side=signal.action,
            entry_price=Decimal(str(signal.indicators["close"])),
            position_size_usd=Decimal(str(position_size)),
            leverage=leverage,
            entry_rsi=signal.indicators.get("rsi"),
            entry_macd_hist=signal.indicators.get("macd_hist"),
            entry_adx=signal.indicators.get("adx"),
            entry_score=signal.score_long
            if signal.action == "LONG"
            else signal.score_short,
            is_open=True,
        )

        if self.paper_mode:
            logging.info(
                f"[PAPER] {signal.action} {self.symbol} @ {signal.indicators['close']:.2f} "
                f"Size:${position_size:.2f} Lev:{leverage}x"
            )
        else:
            logging.info(f"[LIVE] Entry executed: {trade}")
            self._notify_entry(trade, signal)

        return trade

    def _calculate_leverage(self, indicators):
        return self.params.base_leverage

    def _calculate_position_size(self, leverage):
        account_value = 1800
        risk_amount = account_value * self.params.position_size_pct
        position_size = risk_amount * leverage
        return round(position_size, 2)

    def _check_exits(self):
        if not self.paper_mode:
            self._sync_positions_with_bybit()

        open_trades = BybitMechanicalTrade.objects.filter(
            symbol=self.symbol, is_open=True
        )

        for trade in open_trades:
            current_price = self._get_current_price()
            if not current_price:
                continue

            entry = float(trade.entry_price)
            exit_reason = None
            activation_pct = self.params.trailing_stop_activation_pct
            step_pct = self.params.trailing_stop_step_pct

            if trade.side == "LONG":
                sl_price = entry * (1 - self.params.stop_loss_pct)
                tp_price = entry * (1 + self.params.take_profit_pct)
                profit_pct = (current_price - entry) / entry

                if not trade.trailing_stop_activated and profit_pct >= activation_pct:
                    trade.trailing_stop_activated = True
                    trade.trailing_stop_price = Decimal(
                        str(entry * (1 + activation_pct - step_pct))
                    )
                    trade.save()
                    logging.info(
                        f"Trailing stop activated @ {trade.trailing_stop_price}"
                    )

                if trade.trailing_stop_activated and trade.trailing_stop_price:
                    ts_price = float(trade.trailing_stop_price)
                    new_ts = current_price * (1 - step_pct)
                    if new_ts > ts_price:
                        trade.trailing_stop_price = Decimal(str(new_ts))
                        trade.save()
                        logging.info(f"Trailing stop updated to {new_ts:.4f}")

                if current_price <= sl_price:
                    exit_reason = "SL"
                elif current_price >= tp_price:
                    exit_reason = "TP"
                elif trade.trailing_stop_activated and trade.trailing_stop_price:
                    if current_price <= float(trade.trailing_stop_price):
                        exit_reason = "TS"

            else:
                sl_price = entry * (1 + self.params.stop_loss_pct)
                tp_price = entry * (1 - self.params.take_profit_pct)
                profit_pct = (entry - current_price) / entry

                if not trade.trailing_stop_activated and profit_pct >= activation_pct:
                    trade.trailing_stop_activated = True
                    trade.trailing_stop_price = Decimal(
                        str(entry * (1 - activation_pct + step_pct))
                    )
                    trade.save()
                    logging.info(
                        f"Trailing stop activated @ {trade.trailing_stop_price}"
                    )

                if trade.trailing_stop_activated and trade.trailing_stop_price:
                    ts_price = float(trade.trailing_stop_price)
                    new_ts = current_price * (1 + step_pct)
                    if new_ts < ts_price:
                        trade.trailing_stop_price = Decimal(str(new_ts))
                        trade.save()
                        logging.info(f"Trailing stop updated to {new_ts:.4f}")

                if current_price >= sl_price:
                    exit_reason = "SL"
                elif current_price <= tp_price:
                    exit_reason = "TP"
                elif trade.trailing_stop_activated and trade.trailing_stop_price:
                    if current_price >= float(trade.trailing_stop_price):
                        exit_reason = "TS"

            if exit_reason:
                self._execute_exit(trade, current_price, exit_reason)

    def _sync_positions_with_bybit(self):
        try:
            remote_positions = bybit.get_open_positions(self.symbol, category="linear")
        except Exception as e:
            logging.warning(f"Failed to fetch Bybit positions for sync: {e}")
            return

        remote_symbols = {
            p["symbol"] for p in remote_positions if float(p.get("size", 0)) > 0
        }

        db_open = BybitMechanicalTrade.objects.filter(symbol=self.symbol, is_open=True)
        for trade in db_open:
            if trade.symbol not in remote_symbols:
                logging.info(
                    f"Position {trade.symbol} {trade.side} not found on Bybit. "
                    f"Marking as closed (reason=MANUAL)."
                )
                trade.is_open = False
                trade.close_reason = trade.close_reason or "MANUAL"
                trade.closed_at = timezone.now()
                trade.save()

    def _execute_exit(self, trade, exit_price, reason):
        entry = float(trade.entry_price)

        if trade.side == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100 * trade.leverage
        else:
            pnl_pct = (entry - exit_price) / entry * 100 * trade.leverage

        pnl_usd = float(trade.position_size_usd) * pnl_pct / 100

        trade.exit_price = Decimal(str(exit_price))
        trade.close_reason = reason
        trade.pnl_pct = pnl_pct
        trade.pnl_usd = Decimal(str(pnl_usd))
        trade.is_open = False
        trade.closed_at = timezone.now()
        trade.save()

        logging.info(
            f"[{'PAPER' if self.paper_mode else 'LIVE'}] {reason} {trade.side} "
            f"@{exit_price:.2f} PnL:{pnl_pct:.2f}% (${pnl_usd:.2f})"
        )

        if not self.paper_mode:
            self._notify_exit(trade, exit_price, reason)

    def _get_current_price(self):
        try:
            rows = bybit.get_kline(
                self.symbol, interval="1", limit=1, category="linear"
            )
            if rows:
                return float(rows[0][4])
        except Exception as e:
            logging.error(f"Failed to get current price: {e}")
        return None

    def _notify_entry(self, trade, signal):
        try:
            config = TradingConfig.objects.filter(bybit_alert_enabled=True).first()
            if not config:
                return

            lines = [
                f"🔔 Bybit {trade.side} 진입",
                f"- 심볼: {trade.symbol}",
                f"- 진입가: ${float(trade.entry_price):,.2f}",
                f"- 레버리지: {trade.leverage}x",
                f"- 포지션 크기: ${float(trade.position_size_usd):.2f}",
                f"- RSI: {signal.indicators.get('rsi', 0):.1f}",
                f"- MACD hist: {signal.indicators.get('macd_hist', 0):.4f}",
                f"- ADX: {signal.indicators.get('adx', 0):.1f}",
                f"- 손절: ${float(trade.entry_price) * (1 - self.params.stop_loss_pct):,.2f}",
                f"- 익절: ${float(trade.entry_price) * (1 + self.params.take_profit_pct):,.2f}",
            ]

            send_message("\n".join(lines), chat_id=config.telegram_chat_id)
        except Exception as e:
            logging.error(f"Failed to send entry notification: {e}")

    def _notify_exit(self, trade, exit_price, reason):
        try:
            config = TradingConfig.objects.filter(bybit_alert_enabled=True).first()
            if not config:
                return

            reason_map = {
                "SL": "손절",
                "TP": "익절",
                "SIGNAL": "신호반전",
                "MANUAL": "수동",
            }
            reason_text = reason_map.get(reason, reason)
            emoji = "🔴" if reason == "SL" else "🟢"

            lines = [
                f"{emoji} Bybit {trade.side} 청산 ({reason_text})",
                f"- 심볼: {trade.symbol}",
                f"- 진입가: ${float(trade.entry_price):,.2f}",
                f"- 청산가: ${exit_price:,.2f}",
                f"- PnL: {trade.pnl_pct:.2f}% (${float(trade.pnl_usd):.2f})",
                f"- 레버리지: {trade.leverage}x",
            ]

            send_message("\n".join(lines), chat_id=config.telegram_chat_id)
        except Exception as e:
            logging.error(f"Failed to send exit notification: {e}")

    def _log_status(self):
        open_count = BybitMechanicalTrade.objects.filter(
            symbol=self.symbol, is_open=True
        ).count()
        today_pnl = self._calculate_today_pnl()

        logging.info(
            f"Status: {open_count} open positions | Today PnL: ${today_pnl:.2f}"
        )

    def _calculate_today_pnl(self):
        today_trades = BybitMechanicalTrade.objects.filter(
            symbol=self.symbol,
            closed_at__date=timezone.now().date(),
        )
        return sum(t.pnl_usd or 0 for t in today_trades)

    def _save_signal(self, indicators, score_long, score_short, action):
        BybitMechanicalSignal.objects.create(
            symbol=self.symbol,
            timeframe="5m",
            candle_time=timezone.now(),
            close_price=indicators.get("close", 0),
            rsi=indicators.get("rsi"),
            macd=indicators.get("macd"),
            macd_hist=indicators.get("macd_hist"),
            ema20=indicators.get("ema20"),
            ema50=indicators.get("ema50"),
            volume=indicators.get("volume"),
            volume_ma20=indicators.get("volume_ma20"),
            atr=indicators.get("atr"),
            atr_avg=indicators.get("atr_avg"),
            adx=indicators.get("adx"),
            bb_upper=indicators.get("bb_upper"),
            bb_lower=indicators.get("bb_lower"),
            bb_position=indicators.get("bb_position"),
            volume_ratio=indicators.get("volume_ratio"),
            score_long=score_long,
            score_short=score_short,
            conditions_met=max(score_long, score_short) if action else 0,
            action=action,
        )


class BybitSignalData:
    def __init__(self, action, score_long, score_short, indicators):
        self.action = action
        self.score_long = score_long
        self.score_short = score_short
        self.indicators = indicators


class BybitDailyReviewer:
    def __init__(self):
        self.params = self._load_parameters()

    def _load_parameters(self):
        params = BybitMechanicalParameter.objects.first()
        if not params:
            params = BybitMechanicalParameter.objects.create()
        return params

    def run(self, override_hours=None):
        lookback_hours = override_hours or self.params.review_lookback_hours
        since = timezone.now() - timedelta(hours=lookback_hours)

        trades = BybitMechanicalTrade.objects.filter(created__gte=since)
        performance = self._analyze_performance(trades)

        suggestion = self._request_llm_review(performance, lookback_hours)

        if suggestion and suggestion.should_adjust:
            self._create_parameter_proposal(suggestion)

        return {
            "lookback_hours": lookback_hours,
            "total_trades": performance["total_trades"],
            "win_rate": performance["win_rate"],
            "profit_factor": performance["profit_factor"],
            "total_pnl": performance["total_pnl"],
            "should_adjust": suggestion.should_adjust if suggestion else False,
            "reasoning": suggestion.reasoning if suggestion else None,
        }

    def _analyze_performance(self, trades):
        total_trades = trades.count()

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown": 0.0,
                "long_trades": 0,
                "short_trades": 0,
            }

        closed_trades = trades.filter(is_open=False)
        wins = [t for t in closed_trades if (t.pnl_pct or 0) > 0]
        losses = [t for t in closed_trades if (t.pnl_pct or 0) <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (
            win_count / (win_count + loss_count) * 100
            if (win_count + loss_count) > 0
            else 0.0
        )

        total_win = sum(t.pnl_pct or 0 for t in wins)
        total_loss = abs(sum(t.pnl_pct or 0 for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)
        avg_win = total_win / win_count if win_count > 0 else 0.0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(closed_trades)

        long_trades = trades.filter(side="LONG").count()
        short_trades = trades.filter(side="SHORT").count()

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": float(total_pnl),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "open_positions": trades.filter(is_open=True).count(),
        }

    def _calculate_max_drawdown(self, trades):
        if not trades:
            return 0.0

        cumulative = 0
        peak = 0
        max_dd = 0

        for trade in trades.order_by("created"):
            cumulative += trade.pnl_pct or 0
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return max_dd

    def _request_llm_review(self, performance, lookback_hours):
        if performance["total_trades"] < 5:
            logging.info(
                f"Insufficient trades for review: {performance['total_trades']}"
            )
            return None

        current = self.params

        prompt = f"""You are a cryptocurrency trading performance analyst. Review the Bybit mechanical trading system's performance and determine whether parameter adjustments are needed.

[Performance Summary - Last {lookback_hours} hours]
- Total trades: {performance["total_trades"]} (Long: {performance["long_trades"]}, Short: {performance["short_trades"]})
- Win rate: {performance["win_rate"]:.1f}%
- Profit Factor: {performance["profit_factor"]:.2f}
- Total PnL: ${performance["total_pnl"]:.2f}
- Average win: {performance["avg_win"]:.2f}%
- Average loss: -{performance["avg_loss"]:.2f}%
- Max drawdown: {performance["max_drawdown"]:.2f}%
- Open positions: {performance["open_positions"]}

[Current Parameters]
- RSI thresholds: Buy {current.rsi_buy_threshold}, Sell {current.rsi_sell_threshold}
- MACD min histogram: {current.macd_min_histogram}
- Min entry score: {current.min_score_for_entry}
- Min score gap: {current.min_score_gap}
- SL/TP: {current.stop_loss_pct * 100:.1f}% / {current.take_profit_pct * 100:.1f}%
- Position size: {current.position_size_pct * 100:.1f}% of account
- Leverage: {current.base_leverage}x (max {current.max_leverage}x)
- Daily max trades: {current.daily_max_trades}

[Decision Criteria]
- Win rate < 40% OR Profit Factor < 1.2: Parameter adjustment needed
- 3+ consecutive losses: Tighten entry conditions
- Win rate > 60% AND Profit Factor > 2.0: Consider maintaining or relaxing parameters

Respond in JSON format:
1. should_adjust: true/false
2. reasoning: Detailed analysis and rationale (write in Korean)
3. suggested_changes: Only changed parameter values (e.g. {{"rsi_buy_threshold": 30}})
4. expected_improvement: Expected improvement effect (write in Korean)"""

        try:
            from pydantic import BaseModel, Field

            class ParameterSuggestion(BaseModel):
                should_adjust: bool = Field(
                    ..., description="Whether parameters should be adjusted"
                )
                reasoning: str = Field(..., description="Detailed reasoning in Korean")
                suggested_changes: dict = Field(
                    default_factory=dict, description="Specific parameter changes"
                )
                expected_improvement: str = Field(
                    ..., description="Expected improvement in Korean"
                )

            result = invoke_llm(
                "You are a trading performance analyst. Review the data and suggest parameter adjustments.",
                prompt,
                model=ParameterSuggestion,
                structured_output=True,
            )
            logging.info(f"LLM review result: should_adjust={result.should_adjust}")
            return result
        except Exception as e:
            logging.exception(f"LLM review failed: {e}")
            return None

    def _create_parameter_proposal(self, suggestion):
        current = self.params

        changes = suggestion.suggested_changes
        fields = [
            "rsi_buy_threshold",
            "rsi_sell_threshold",
            "macd_min_histogram",
            "adx_min_threshold",
            "rsi_weight",
            "macd_weight",
            "ema_weight",
            "volume_weight",
            "min_score_for_entry",
            "min_score_gap",
            "stop_loss_pct",
            "take_profit_pct",
            "max_positions",
            "position_size_pct",
            "base_leverage",
            "max_leverage",
            "leverage_atr_multiplier",
            "entry_cooldown_minutes",
            "daily_max_trades",
        ]

        new_values = {
            field: changes.get(field, getattr(current, field)) for field in fields
        }

        new_params = BybitMechanicalParameter.objects.create(
            review_lookback_hours=current.review_lookback_hours,
            llm_reasoning=suggestion.reasoning,
            **new_values,
        )

        logging.info(f"Created new parameter proposal (id={new_params.id})")

        try:
            import json

            config = TradingConfig.objects.filter(bybit_alert_enabled=True).first()
            if config:
                changes_json = json.dumps(
                    suggestion.suggested_changes, ensure_ascii=False, indent=2
                )
                send_message(
                    f"📋 Bybit 파라미터 조정 제안\n\n{suggestion.reasoning}\n\n"
                    f"제안된 변경:\n{changes_json}\n\n"
                    f"예상 효과: {suggestion.expected_improvement}\n\n"
                    f"관리자 페이지에서 적용 여부를 결정하세요.",
                    chat_id=config.telegram_chat_id,
                )
        except Exception as e:
            logging.error(f"Failed to send proposal notification: {e}")

        return new_params
