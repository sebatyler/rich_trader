import json
import logging
import re
import time
from datetime import timedelta
from decimal import Decimal
from typing import Optional
from typing import Literal

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
from core.indicators import ema as calc_ema
from core.indicators import macd as calc_macd
from core.indicators import rsi as calc_rsi
from core.indicators import volume_ma as calc_volume_ma
from core.llm import invoke_gemini_search
from core.llm import invoke_llm
from core.parameter_optimization import create_optimization_payload
from core.parameter_optimization import request_optimized_parameters
from core.telegram import send_message
from core.utils import dict_at
from core.utils import dict_omit
from core.utils import format_currency
from core.utils import format_quantity
from trading.models import AlgorithmParameter
from trading.models import AutoTrading
from trading.models import BybitSignal
from trading.models import Portfolio
from trading.models import Trading
from trading.models import TradingConfig
from trading.models import UpbitTrading

from .models import CryptoListing


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
    cond_trend = ind_5m["close"] > ind_5m["ema20"] and ind_5m["ema20"] >= ind_5m["ema50"]
    cond_vol = ind_5m["volume"] >= ind_5m["volume_ma20"]
    cond_15m_trend = ind_15m["ema20"] >= ind_15m["ema50"]
    return cond_rsi and cond_macd and cond_trend and cond_vol and cond_15m_trend


def _rule_based_scalp_long_signal(ind_3m: dict, ind_5m: dict, ind_15m: dict) -> bool:
    cond_rsi = ind_3m["rsi"] > 45 and ind_5m["rsi"] >= 45
    cond_macd = ind_3m["macd_hist"] > 0
    cond_trend = ind_3m["close"] > ind_3m["ema20"] >= ind_3m["ema50"]
    cond_vol = ind_3m["volume"] >= ind_3m["volume_ma20"]
    cond_htf_trend = ind_5m["ema20"] >= ind_5m["ema50"] and ind_15m["ema20"] >= ind_15m["ema50"]
    return cond_rsi and cond_macd and cond_trend and cond_vol and cond_htf_trend


def _rule_based_scalp_short_signal(ind_3m: dict, ind_5m: dict, ind_15m: dict) -> bool:
    cond_rsi = ind_3m["rsi"] < 55 and ind_5m["rsi"] <= 55
    cond_macd = ind_3m["macd_hist"] < 0
    cond_trend = ind_3m["close"] < ind_3m["ema20"] <= ind_3m["ema50"]
    cond_vol = ind_3m["volume"] >= ind_3m["volume_ma20"]
    cond_htf_trend = ind_5m["ema20"] <= ind_5m["ema50"] and ind_15m["ema20"] <= ind_15m["ema50"]
    return cond_rsi and cond_macd and cond_trend and cond_vol and cond_htf_trend


def _format_telegram_message(symbol: str, side: str, tf3: dict, tf5: dict, tf15: dict, decision: dict) -> str:
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
                "macd_3m": {"macd": ind3["macd"], "signal": ind3["macd_signal"], "hist": ind3["macd_hist"]},
                "macd_5m": {"macd": ind5["macd"], "signal": ind5["macd_signal"], "hist": ind5["macd_hist"]},
                "ema_3m": {"ema20": ind3["ema20"], "ema50": ind3["ema50"]},
                "ema_5m": {"ema20": ind5["ema20"], "ema50": ind5["ema50"]},
                "ema_15m": {"ema20": ind15["ema20"], "ema50": ind15["ema50"]},
                "vol_ma_3m": {"vol": ind3["volume"], "vol_ma20": ind3["volume_ma20"]},
                "atr_3m": ind3["atr"],
            },
            "last_closed_candle": {"time": df3.iloc[-1]["time"].isoformat(), "close": ind3["close"]},
            "fees": {"derivatives_taker": 0.00055, "derivatives_maker": 0.0002},
            "rule_based": {"long": should_long, "short": should_short},
        }

        class BybitDecision(BaseModel):
            trade_signal: bool = Field(..., description="Whether to enter a trade now")
            side: Literal["LONG","SHORT"] = Field(..., description="Trade direction")
            confidence: float = Field(..., description="The confidence 0-1")
            reason: str = Field(..., description="The reason in Korean (<=2 sentences)")
            entry_price: float = Field(..., description="The entry price")
            stop_loss: float = Field(..., description="The stop loss")
            take_profit: float = Field(..., description="The take profit reachable in minutes")
            expected_profit_pct: float = Field(..., description="Net expected profit % at 1x, fees included")
            recommended_leverage: Literal[10,25,50] = Field(..., description="Pick 10,25,50 only")
            few_minutes_profitable: bool = Field(..., description="Likely to realize within next 1-3 3m candles")

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
        try:
            content = json.dumps(payload, ensure_ascii=False)
            decision_obj = invoke_llm(
                system,
                content,
                model=BybitDecision,
                structured_output=True,
                template_format="jinja2",
            )
            decision = decision_obj.model_dump()
        except Exception as e:
            # Fallback conservative rule-based decision
            try:
                close = ind3["close"]
                atr = ind3["atr"]
                # Choose side by signal
                if should_long and not should_short:
                    side = "LONG"
                elif should_short and not should_long:
                    side = "SHORT"
                else:
                    side = "LONG" if ind3["macd_hist"] >= 0 else "SHORT"
                if side == "LONG":
                    entry = close
                    stop = max(close - 0.5 * atr, 0)
                    take = close + 1.0 * atr
                    rr = (take - entry) / (entry - stop) if (entry - stop) > 0 else 0
                    expected = ((take / entry) - 1 - 0.00055 - 0.00055) * 100
                else:
                    entry = close
                    stop = close + 0.5 * atr
                    take = max(close - 1.0 * atr, 0)
                    rr = (entry - take) / (stop - entry) if (stop - entry) > 0 else 0
                    expected = ((entry / take) - 1 - 0.00055 - 0.00055) * 100 if take > 0 else -1
                # leverage heuristic
                stop_dist_pct = abs(entry - stop) / entry * 100 if entry > 0 else 100
                if stop_dist_pct < 0.35:
                    lev = 50
                elif stop_dist_pct < 0.8:
                    lev = 25
                else:
                    lev = 10
                decision = {
                    "trade_signal": rr >= 1.5 and expected >= 0.1,
                    "side": side,
                    "confidence": 0.55,
                    "reason": "보수적 규칙 기반 백업 결정입니다.",
                    "entry_price": float(entry),
                    "stop_loss": float(stop),
                    "take_profit": float(take),
                    "expected_profit_pct": float(expected),
                    "recommended_leverage": lev,
                    "few_minutes_profitable": bool(ind3["macd_hist"] * (1 if side == "LONG" else -1) > 0 and ind3["volume"] >= ind3["volume_ma20"]),
                    "error": str(e),
                }
            except Exception as ee:
                decision = {"error": f"{e} | fallback_failed: {ee}"}

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
                buy_signal=bool(decision.get("trade_signal", False)),
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
            should_notify = bool(decision.get("trade_signal")) and isinstance(exp, (int, float)) and exp >= 0.1
            if should_notify:
                text = _format_telegram_message(symbol, decision.get("side","LONG"), ind3, ind5, ind15, decision)
                for cfg in configs:
                    # double-check enabled and membership
                    if cfg.bybit_alert_enabled and symbol in (cfg.bybit_target_coins or []):
                        chat_id = cfg.telegram_chat_id
                        if chat_id:
                            try:
                                send_message(text, chat_id=chat_id, is_markdown=False)
                            except Exception:
                                logging.exception(f"Failed to send Telegram message for {symbol} to chat_id={chat_id}")

        results.append({
            "symbol": symbol,
            "should_long": should_long,
            "should_short": should_short,
            "decision": decision,
        })

    return results


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class Recommendation(BaseStrippedModel):
    action: str = Field(..., description="The action to take (BUY or SELL)")
    symbol: str = Field(..., description="The symbol of the cryptocurrency")
    amount: Optional[int] = Field(default=None, description="The amount of the cryptocurrency to buy in KRW")
    quantity: Optional[float] = Field(default=None, description="The quantity of the cryptocurrency to sell")
    limit_price: Optional[float] = Field(default=None, description="The limit price for the order")
    reason: str = Field(..., description="The reason for the recommendation")


class MultiCryptoRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The analysis scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    recommendations: list[Recommendation] = Field(..., description="List of recommended cryptocurrency trades")


def collect_crypto_data(symbol: str, start_date: str, news_count: int = 10, from_upbit: bool = False):
    """특정 암호화폐의 모든 관련 데이터를 수집합니다."""
    if from_upbit:
        tickers = upbit.get_ticker(symbol)
        ticker = tickers[0]
        crypto_price = ticker["trade_price"]
    else:
        ticker = coinone.get_ticker(symbol)
        crypto_price = (float(ticker["best_asks"][0]["price"]) + float(ticker["best_bids"][0]["price"])) / 2

    crypto_data = crypto.get_quotes(symbol)

    input_data = dict(
        ticker,
        circulating_supply=crypto_data["circulating_supply"],
        max_supply=crypto_data["max_supply"],
        total_supply=crypto_data["total_supply"],
        **crypto_data["quote"]["KRW"],
        current_price=crypto_price,
    )

    # 과거 데이터 수집
    historical_data = crypto.get_historical_data(symbol, "KRW", 30)
    df = pd.DataFrame(historical_data)
    df = df.drop(columns=["conversionType", "conversionSymbol"])
    crypto_data_csv = df.to_csv(index=False)

    # 네트워크 데이터 (비트코인만)
    network_stats_csv = ""
    if symbol == "BTC":
        network_stats = crypto.get_network_stats()
        df = pd.DataFrame(network_stats, index=[0])
        network_stats_csv = df.to_csv(index=False)

    # 뉴스 데이터
    if settings.DEBUG:
        crypto_news_csv = ""
    else:
        crypto_news = crypto.fetch_news(start_date, symbol, news_count)
        df = pd.DataFrame(crypto_news)
        df = df[["source", "title", "description", "publishedAt", "content"]]
        df["source"] = df["source"].apply(lambda x: x["name"])
        crypto_news_csv = df.to_csv(index=False)

    return {
        "symbol": symbol,
        "input_data": input_data,
        "crypto_data_csv": crypto_data_csv,
        "network_stats_csv": network_stats_csv,
        "crypto_news_csv": crypto_news_csv,
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
            rf"^({symbol}_(json_data|crypto_data_csv|network_stats_csv|crypto_news_csv|balance_json|market_json))",
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
                f"{symbol}_json_data": json.dumps(data["input_data"]),
                f"{symbol}_market_json": json.dumps(market),
                f"{symbol}_crypto_data_csv": data["crypto_data_csv"],
                f"{symbol}_crypto_news_csv": data["crypto_news_csv"],
            }
        )
        if data["network_stats_csv"]:  # BTC인 경우
            kwargs[f"{symbol}_network_stats_csv"] = data["network_stats_csv"]

    krw_balance = int(float(balances["KRW"]["available"] or 0))
    prompt = f"""You are a crypto trading advisor who is aggressive yet risk-aware. You have access to:
 - Real-time data, historical prices, volatility, news, sentiment
 - Recent trading history in CSV format
 - KRW balance: {krw_balance:,} KRW
 - Total coin value: {total_coin_value:,} KRW
 - Total portfolio value: {total_coin_value + krw_balance:,} KRW
 - Min trade: {trading_config.min_trade_amount:,} KRW, step: {trading_config.step_amount:,} KRW

Key Rules (CRITICAL - FOLLOW EXACTLY):
1) Trade Recommendation Count Rules:
   - Recommend exactly {trading_config.min_coins} to {trading_config.max_coins} trades, or 0 if no good opportunities
   - NEVER exceed {trading_config.max_coins} trades
   - NEVER recommend both BUY and SELL for the same coin
   - Each coin can appear only once in recommendations
   - Consider recent trading history to avoid frequent trading of the same coin

2) BUY constraints:
   - amount ≥ {trading_config.min_trade_amount}, multiple of {trading_config.step_amount}
   - Single BUY ≤ 30% of available KRW, total BUY ≤ 50% of KRW
   - Only recommend BUY if strong upward momentum and positive news
   - Avoid buying coins that were recently sold at a loss

3) SELL constraints:
   - quantity must respect exchange increments (qty_unit) and min_qty~max_qty range
   - Consider partial selling if large holdings, to manage risk and slippage
   - limit_price ~ 0.1~0.3% below current for execution
   - Only recommend SELL if downward trend or risk mitigation needed
   - Consider profit/loss from recent trades of the same coin

4) Fees & Profit:
   - Fee: 0.02% each trade (0.04% round-trip)
   - Price must move ≥ 0.06% to surpass fees (add ~0.02% safety margin)
   - Track cumulative fees from recent trades

5) Risk & Volatility:
   - Avoid risking >2~3% of total portfolio on a single trade
   - High volatility => smaller positions, possibly more diversification
   - Factor in recent news/sentiment for short-term moves
   - Consider recent trading performance of each coin

6) Final KRW Ratio:
   - After ALL recommended BUY/SELL are done, aim for 10%~30% of total portfolio in KRW
   - If below 10% or above 30%, explain (e.g., strong bullish/bearish outlook, waiting for better entries)

7) Recent Trading Analysis:
   - Review recent trades from CSV data
   - Consider win/loss ratio for each coin
   - Avoid overtrading by checking trade frequency
   - Factor in realized profits/losses

Output must be valid YAML with these sections:
```yaml
scratchpad: |
  [시장 상황과 최근 거래 분석 (한국어). 핵심 포인트만 3-4줄로 작성]

reasoning: |
  [매매 전략 설명 (한국어). 핵심 포인트만 3-4줄로 작성]

recommendations:
  - action: "BUY"    # or "SELL"
    symbol: "BTC"
    amount: 500000   # (int or null) for BUY only
    quantity: null   # (float or null) for SELL only
    limit_price: null  # (int or null) for SELL only
    reason: "핵심적인 매매 사유 1-2줄로 작성"
```

Rules:
1. Strictly follow the YAML structure above
2. scratchpad and reasoning MUST use multiline string format with | operator and consistent indentation
3. Keep total length of scratchpad + reasoning < 2000 chars
4. Each line should be a complete, meaningful statement
5. Use simple, clear Korean language
6. No repetition between sections
7. Double-check that recommendations follow all trade recommendation count and constraint rules
8. No extra fields. No extra lines outside the YAML
"""
    if settings.DEBUG:
        with open(f"tmp/{trading_config.user.email.split('@')[0]}.txt", "w") as f:
            f.write(prompt)
            f.write(all_data)
            f.write(json.dumps(kwargs))

    return invoke_llm(prompt, all_data, model=MultiCryptoRecommendation, with_fallback=with_fallback, **kwargs)


def get_rebalance_recommendation(
    crypto_data_list: list[dict], indices_csv: str, balances: dict[str, dict], total_coin_value: int
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
        with open(f"tmp/rebalance.txt", "w") as f:
            f.write(prompt)
            f.write(all_data)
            f.write(json.dumps(kwargs))

    return invoke_llm(prompt, all_data, with_anthropic=True, **kwargs)


def send_trade_result(trading: Trading, balances: dict, chat_id: str):
    """거래 결과를 확인하고 텔레그램 메시지를 전송합니다."""
    symbol = trading.coin
    quantity = Decimal(trading.executed_qty or 0)
    amount = int(quantity * (trading.average_executed_price or 0))

    message_lines = [f"{trading.side}: {format_quantity(quantity)} {symbol} ({amount:,} 원)"]
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
        message_lines.append(f"주문 취소됨! 주문하는게 좋다고 판단하면 직접 주문하세요. {trading.side} / {order}")

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


def execute_trade(user, recommendation: Recommendation, crypto_data: dict, chat_id: str) -> dict:
    """거래를 실행하고 결과를 처리합니다."""
    action = recommendation.action
    symbol = recommendation.symbol
    crypto_price = crypto_data["input_data"]["current_price"]

    logging.info(f"{recommendation=}")

    if settings.DEBUG:
        return

    if action == "BUY":
        amount = recommendation.amount
        if not amount:
            raise ValueError("amount is required for buy order")

        order = coinone.buy_ticker(symbol, amount)
    elif action == "SELL":
        quantity = recommendation.quantity
        limit_price = recommendation.limit_price
        if not quantity:
            raise ValueError("quantity is required for sell order")

        order = coinone.sell_ticker(symbol, quantity, limit_price)
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
        limit_price=recommendation.limit_price,
        crypto_price=crypto_price,
        order_detail=order_detail,
        chat_id=chat_id,
        reason=recommendation.reason,
    )


def auto_trading():
    """암호화폐 매매 프로세스를 실행합니다."""
    # 오늘 날짜와 한 달 전 날짜 설정
    end_date = timezone.localdate()
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # 시장 지표 데이터 가져오기
    indices_data_csv = crypto.get_market_indices(start_date)

    # 전체 종목 정보 가져오기 (qty_unit 정보 포함)
    markets = coinone.get_markets()

    # 활성화된 트레이딩 설정에서 모든 target_coins를 가져와서 중복 제거
    active_configs = TradingConfig.objects.filter(is_active=True)
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
        recent_trades_csv = Trading.get_recent_trades_csv(user=config.user)

        # 해당 유저의 target_coins에 대한 데이터만 필터링하고 현재 잔고 가치 계산
        user_crypto_data = {}
        total_coin_value = 0
        for symbol in config.target_coins:
            if symbol in crypto_data_dict:
                data = dict(crypto_data_dict[symbol])
                balance = balances.get(symbol)
                if balance:
                    current_value = float(balance.get("available") or 0) * data["input_data"]["current_price"]
                    data["input_data"]["current_value"] = current_value
                    total_coin_value += current_value
                user_crypto_data[symbol] = data

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
                    with_fallback=i > 0,
                )
                break
            except Exception as e:
                logging.warning(e)
                exc = e

        if not result and exc:
            logging.exception(f"Error getting multi recommendation for {config.user}: {exc}")
            continue

        # 분석 결과 전송
        send_message(
            f"```\n코인 분석:\n{result.scratchpad}\n\n{result.reasoning}```",
            chat_id=chat_id,
            is_markdown=True,
        )

        final_balances = None

        # 추천받은 거래 실행
        for recommendation in result.recommendations:
            symbol = recommendation.symbol
            crypto_data = user_crypto_data[symbol]

            try:
                execute_trade(
                    config.user,
                    recommendation=recommendation,
                    crypto_data=crypto_data,
                    chat_id=chat_id,
                )
            except Exception as e:
                logging.exception(f"Error executing trade for {symbol}: {e}")

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
                current_value = float(balance.get("quantity", 0)) * crypto_data["input_data"]["current_price"]
                crypto_data["input_data"]["current_value"] = current_value
                total_coin_value += current_value
            crypto_data_dict[symbol] = crypto_data
        except Exception as e:
            logging.exception(f"Failed to collect data for {symbol}: {e}")
            continue

    config = TradingConfig.objects.filter(user__is_superuser=True).first()
    chat_id = config.telegram_chat_id

    # 해당 유저의 target_coins에 대한 데이터만 필터링
    crypto_data = {symbol: crypto_data_dict[symbol] for symbol in target_coins if symbol in crypto_data_dict}

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
        text_list.extend([f"{i}. {coin['name']} ({coin['symbol']}) ${coin['last_price']:.4f}"])
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
    major_coins = {"BTC", "ETH", "XRP", "SOL"}
    # TODO: django constance
    return {coin: 20_000 if coin in major_coins else 10_000 for coin in coins}


def _buy_upbit_coins():
    data = upbit.get_balance_data()
    balances, total_value, krw_value = dict_at(data, "balances", "total_value", "krw_value")

    # 총 자산이 4억 원 이상이면 구매 중지
    if total_value >= 400_000_000:
        return

    coins = {balance["symbol"].split(".")[0] for balance in balances}
    coin_amounts = get_coin_amounts(coins)

    # 원화 잔고가 코인 구매에 필요한 금액보다 적으면 구매 중지
    required_krw = sum(coin_amounts.values())
    logging.info(f"{required_krw=:,} {krw_value=:,.0f}")
    if krw_value < required_krw:
        return

    # 코인 구매
    for coin, amount in coin_amounts.items():
        last_trading = next(
            (
                UpbitTrading.objects.filter(coin=coin, is_dca=is_dca).order_by("-created").first()
                for is_dca in (False, True)
            ),
            None,
        )
        if not last_trading:
            continue

        last_buy_price = last_trading.average_price
        last_buy_at = last_trading.created

        last_candle = upbit.get_candles(coin, count=1)[0]
        last_price = Decimal(last_candle["trade_price"])

        # 마지막 매수한지 1시간 이상 지났고 1% 이상 하락했을 때만 구매
        price_change = (last_price - last_buy_price) / last_buy_price * 100
        should_buy = price_change <= -1 and last_buy_at < timezone.now() - timedelta(hours=1)
        logging.info(
            f"{coin}: {should_buy=} {format_quantity(last_price)} <- {format_quantity(last_buy_price)} ({price_change:.2f}%) {last_buy_at}"
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
    coin_amounts = get_coin_amounts(coins)

    # 원화 잔고가 코인 구매에 필요한 금액보다 적으면 구매 중지
    required_krw = sum(coin_amounts.values())
    logging.info(f"{required_krw=:,} {krw_value=:,.0f}")
    if krw_value < required_krw:
        return

    today = timezone.localdate()

    # 코인 구매
    for coin, amount in coin_amounts.items():
        # 오늘 이미 매수한 코인이면 매수 안함
        if UpbitTrading.objects.filter(coin=coin, is_dca=True, created__date=today).exists():
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
        coinone.init(access_key=self.config.coinone_access_key, secret_key=self.config.coinone_secret_key)
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
        algo_param = AlgorithmParameter.objects.filter(user=self.user).order_by("-id").first()
        if not algo_param:
            algo_param = AlgorithmParameter.objects.create(user=self.user)
        if len(prices) < max(algo_param.rsi_period, algo_param.bollinger_period):
            return
        prices.reverse()
        rsi_values = ta.calculate_rsi(prices, period=algo_param.rsi_period)
        latest_rsi = rsi_values[-1]
        middle, upper, lower = ta.calculate_bollinger_bands(
            prices, period=algo_param.bollinger_period, num_std=float(algo_param.bollinger_std)
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
            total_bid_volume / (total_bid_volume + total_ask_volume) if total_bid_volume + total_ask_volume > 0 else 0.5
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
                logging.info(f"STOP LOSS 쿨타임 적용중: {elapsed:.1f}분 경과, {cooldown}분 대기 필요")
                return

        # --- 분할매수/추가매수 로직 (최근 매도 이후 매수 기준) ---
        buy_ratio = getattr(algo_param, "buy_chunk_ratio", 0.5)
        min_trade_amount = float(btc_market.get("min_order_amount", 0))
        buy_amount = max(min_trade_amount, krw_available * algo_param.max_krw_buy_ratio * buy_ratio)
        max_additional_buys = getattr(algo_param, "max_additional_buys", 2)
        # 1) 가장 최근 SELL(매도) AutoTrading의 id 구하기
        last_sell = AutoTrading.objects.filter(trading__side="SELL", trading__coin="BTC").order_by("-created").first()
        if last_sell:
            # 2) 그 id 이후의 BUY(매수) AutoTrading 중 첫 번째는 최초매수, 그 이후만 추가매수로 카운트
            buys_after_sell = list(
                AutoTrading.objects.filter(trading__side="BUY", trading__coin="BTC", id__gt=last_sell.id).order_by(
                    "created"
                )
            )
        else:
            buys_after_sell = list(
                AutoTrading.objects.filter(trading__side="BUY", trading__coin="BTC").order_by("created")
            )
        # 최초매수 이후의 추가매수 카운트
        additional_buy_count = max(0, len(buys_after_sell) - 1)
        add_buy_allowed = (
            btc_available > 0
            and additional_buy_count < max_additional_buys
            and latest_rsi <= getattr(algo_param, "add_buy_rsi_threshold", 20.0)
            and (current_price <= lower + getattr(algo_param, "add_buy_bollinger_band", -1.5) * (upper - middle) / 2)
        )
        # 신규매수 or 추가매수
        if signal == "BUY" and krw_available >= self.config.min_trade_amount:
            if (
                (latest_rsi < algo_param.buy_rsi_threshold or current_price <= lower)
                and buy_pressure >= algo_param.buy_pressure_threshold
                and (profit_rate <= algo_param.buy_profit_rate or btc_available == 0 or add_buy_allowed)
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
        Trading.objects.filter(user=user, coin="BTC", auto_tradings__isnull=False).order_by("-id")[:50].values()
    )
    recent_trades = [dict_omit(trade, "status") for trade in recent_trades]
    # 현재 시점의 포트폴리오를 생성하여 export
    coinone.init(access_key=config.coinone_access_key, secret_key=config.coinone_secret_key)
    balances = coinone.get_balances()
    balances = {
        k: v
        for k, v in balances.items()
        if k == "KRW" or (float(v.get("available") or 0) > 0 and float(v.get("average_price") or 0) > 0)
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
