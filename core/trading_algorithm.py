"""
매매 알고리즘 구현 모듈

이 모듈은 기술적 지표 계산, 매매 신호 생성 및 손절/이익 실현 조건 판단 기능을 제공합니다.
"""

import numpy as np


def calculate_rsi(prices, period=14):
    """
    주어진 가격 데이터(prices)를 이용하여 RSI(상대 강도 지수)를 계산합니다.

    :param prices: 가격 데이터 리스트
    :param period: 계산 기간 (기본: 14)
    :return: RSI 값 리스트
    """
    if len(prices) < period + 1:
        raise ValueError("가격 데이터의 길이가 충분하지 않습니다.")

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.convolve(gains, np.ones(period) / period, mode="valid")
    avg_loss = np.convolve(losses, np.ones(period) / period, mode="valid")

    rs = avg_gain / (avg_loss + 1e-10)  # 0으로 나누는 것을 방지
    rsi = 100 - (100 / (1 + rs))

    # 입력 길이와 맞추기 위해 초기 값을 NaN으로 패딩
    rsi_full = [float("nan")] * period + list(rsi)
    return rsi_full


def calculate_bollinger_bands(prices, period=20, num_std=2):
    """
    주어진 가격 데이터(prices)를 바탕으로 Bollinger Bands를 계산합니다.

    :param prices: 가격 데이터 리스트
    :param period: 계산 기간 (기본: 20)
    :param num_std: 표준편차 배수 (기본: 2)
    :return: (중심선, 상한선, 하한선)
    """
    if len(prices) < period:
        raise ValueError("가격 데이터의 길이가 충분하지 않습니다.")

    prices_array = np.array(prices[-period:])
    middle = np.mean(prices_array)
    std = np.std(prices_array)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def generate_trade_signal(buy_rsi_threshold, sell_rsi_threshold, current_price, rsi, upper_band, lower_band):
    """
    RSI와 Bollinger Bands를 이용하여 매매 신호(BUY, SELL, HOLD)를 생성합니다.

    :param current_price: 현재 가격
    :param rsi: 현재 RSI 값
    :param upper_band: Bollinger 상한선
    :param lower_band: Bollinger 하한선
    :return: 거래 신호 ("BUY", "SELL", "HOLD")
    """
    if rsi < buy_rsi_threshold or current_price < lower_band:
        return "BUY"
    elif rsi > sell_rsi_threshold or current_price > upper_band:
        return "SELL"
    else:
        return "HOLD"


def check_stop_loss_take_profit(entry_price, current_price, stop_loss_pct, take_profit_pct):
    """
    현재 가격과 진입 가격을 기반으로 손절/이익 실현 여부를 판단합니다.

    :param entry_price: 진입 가격
    :param current_price: 현재 가격
    :param stop_loss_pct: 손절 퍼센트 (예: 0.05면 5% 하락 시 손절)
    :param take_profit_pct: 이익 실현 퍼센트 (예: 0.10이면 10% 상승 시 이익 실현)
    :return: 'STOP LOSS', 'TAKE PROFIT', 또는 'HOLD'
    """
    if current_price <= entry_price * (1 - stop_loss_pct):
        return "STOP LOSS"
    elif current_price >= entry_price * (1 + take_profit_pct):
        return "TAKE PROFIT"
    else:
        return "HOLD"
