import hashlib
import os
import time
import uuid
from collections import defaultdict
from decimal import Decimal
from urllib.parse import unquote
from urllib.parse import urlencode

import jwt
import requests

from core.utils import dict_omit

access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")
origin = "https://api.upbit.com"


def _get_headers(params: dict = None) -> dict:
    """공통 헤더를 생성하는 헬퍼 함수"""
    payload = {
        "access_key": access_key,
        "nonce": str(uuid.uuid4()),
    }

    if params:
        query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")
        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload.update(
            query_hash=query_hash,
            query_hash_alg="SHA512",
        )

    jwt_token = jwt.encode(payload, secret_key)
    return {
        "Authorization": f"Bearer {jwt_token}",
    }


def _request(endpoint: str, method: str = "GET", params: dict = None) -> dict:
    """API 요청을 처리하는 공통 함수"""
    headers = _get_headers(params)
    response = requests.request(
        method=method,
        url=f"{origin}{endpoint}",
        headers=headers,
        params=params,
    )
    return response.json()


def get_balances() -> dict:
    """계좌 잔고 조회"""
    return _request("/v1/accounts")


def get_closed_orders() -> dict:
    """완료된 주문 조회"""
    params = {"state": "done"}
    return _request("/v1/orders/closed", params=params)


def get_withdraws(page: int = 1) -> dict:
    """출금 내역 조회"""
    params = {"state": "DONE", "page": page}
    return _request("/v1/withdraws", params=params)


def get_deposits(currency: str = "KRW", page: int = 1) -> dict:
    """입금 내역 조회"""
    params = {"page": page}
    if currency:
        params["currency"] = currency
    return _request("/v1/deposits", params=params)


def get_staking_coins():
    """스테이킹 코인 조회"""
    stakings = defaultdict(Decimal)
    page = 1

    while True:
        withdraws = get_withdraws(page=page)
        if not withdraws:
            break

        for withdraw in withdraws:
            if withdraw["transaction_type"] == "internal" and withdraw["txid"].startswith("staking"):
                stakings[withdraw["currency"]] += Decimal(withdraw["amount"])

        page += 1

    page = 1
    while True:
        deposits = get_deposits(currency=None, page=page)
        if not deposits:
            break

        for deposit in deposits:
            if deposit["transaction_type"] == "internal" and deposit["txid"].startswith("unstaking"):
                stakings[deposit["currency"]] -= Decimal(deposit["amount"])

        page += 1

    return {k: float(v) for k, v in stakings.items()}


def get_available_balances() -> dict:
    """사용 가능한 잔고 조회"""
    balances = {}
    for balance in get_balances():
        symbol = balance["currency"]
        if symbol == "KRW" or float(balance["avg_buy_price"]):
            balances[symbol] = {
                "quantity": f"{Decimal(balance['balance']) + Decimal(balance['locked']):f}",
                "avg_buy_price": balance["avg_buy_price"],
            }

    for symbol, amount in get_staking_coins().items():
        balances[f"{symbol}.S"] = {"quantity": amount, "is_staking": True}

    return balances


def get_ticker(ticker):
    while True:
        data = _request("/v1/ticker", params={"markets": f"KRW-{ticker}"})
        if data and isinstance(data, dict) and data.get("name") == "too_many_requests":
            time.sleep(0.1)
            continue

        break

    return [dict_omit(row, "market") for row in data]


def buy_coin(symbol: str, amount: int):
    """코인 매수"""
    params = {
        "market": f"KRW-{symbol}",
        "side": "bid",
        "ord_type": "price",
        "price": amount,
    }
    return _request("/v1/orders", method="POST", params=params)


def get_candles(symbol: str, count: int = 60):
    """분단위 캔들 조회"""
    return _request(
        "/v1/candles/minutes/1",
        params={"market": f"KRW-{symbol}", "count": count},
    )


def get_closed_orders(symbol: str, count: int = 20):
    """완료된 주문 조회"""
    return _request("/v1/orders/closed", params={"market": f"KRW-{symbol}", "count": count})


def get_order_detail(uuid: str, add_buy_price: bool = True):
    """주문 상세 조회"""
    result = _request("/v1/order", params={"uuid": uuid})

    if add_buy_price:
        total_volume = 0
        total_value = 0
        for trade in result["trades"]:
            volume = Decimal(trade["volume"])
            value = volume * Decimal(trade["price"])
            total_volume += volume
            total_value += value

        result["avg_buy_price"] = float(total_value / total_volume) if total_value and total_volume else None

    return result


def get_balance_data():
    # 업비트 잔고 조회
    balances = get_available_balances()

    # 총 자산 가치 계산을 위해 현재가 조회
    total_coin_value = 0
    krw = balances.pop("KRW", {})
    krw_value = float(krw.get("quantity", 0))
    balance_list = []
    tickers = {}

    for symbol_raw, balance in balances.items():
        symbol = symbol_raw.split(".")[0]

        # 현재가 조회
        ticker = tickers.get(symbol) or get_ticker(symbol)
        if ticker:
            tickers[symbol] = ticker
            current_price = ticker[0]["trade_price"]
            quantity = float(balance["quantity"])
            value = quantity * current_price

            if value >= 5000:
                balance["current_price"] = current_price
                balance["value"] = value
                balance["symbol"] = symbol
                total_coin_value += value
                balance_list.append(balance)

    total_value = total_coin_value + krw_value

    for balance in balance_list:
        balance["weight"] = balance["value"] / total_value * 100

    sorted_balances = sorted(balance_list, key=lambda x: x["value"], reverse=True)

    return {
        "balances": sorted_balances,
        "total_coin_value": total_coin_value,
        "krw_value": krw_value,
        "krw_weight": krw_value / total_value * 100,
        "total_value": total_value,
    }
