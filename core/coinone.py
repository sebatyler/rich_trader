import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime
from datetime import timedelta
from typing import Optional

import requests
from pydantic import BaseModel

from django.utils import timezone

ACCESS_TOKEN = None
SECRET_KEY = None


def init(access_key, secret_key):
    global ACCESS_TOKEN, SECRET_KEY
    ACCESS_TOKEN = access_key
    SECRET_KEY = bytes(secret_key, "utf-8")


def get_encoded_payload(payload):
    payload["nonce"] = str(uuid.uuid4())

    dumped_json = json.dumps(payload)
    encoded_json = base64.b64encode(bytes(dumped_json, "utf-8"))
    return encoded_json


def get_signature(encoded_payload):
    signature = hmac.new(SECRET_KEY, encoded_payload, hashlib.sha512)
    return signature.hexdigest()


def get_response(action, method="post", payload=None, query=None, public=False):
    url = f"https://api.coinone.co.kr/{action.lstrip('/')}"

    headers = {"Accept": "application/json"}
    if not public:
        encoded_payload = get_encoded_payload(payload)
        headers.update(
            {
                "X-COINONE-PAYLOAD": encoded_payload,
                "X-COINONE-SIGNATURE": get_signature(encoded_payload),
            }
        )

    response = requests.request(method, url, headers=headers, json=payload, params=query)
    return response.json()


def get_balances():
    data = get_response(
        action="/v2.1/account/balance/all",
        payload={"access_token": ACCESS_TOKEN},
    )
    return {balance["currency"]: balance for balance in data["balances"]}


def get_ticker(ticker):
    return get_response(f"/public/v2/ticker_new/KRW/{ticker}", method="get", public=True)["tickers"][0]


class OrderResponse(BaseModel):
    result: str
    error_code: str
    error_msg: Optional[str] = None
    order_id: Optional[str] = None


def _order(ticker, side, amount=None, quantity=None, limit_price=None) -> OrderResponse:
    payload = {
        "access_token": ACCESS_TOKEN,
        "quote_currency": "KRW",
        "target_currency": ticker,
        "type": "MARKET",
        "side": side,
    }
    if side == "BUY":
        if not amount:
            raise ValueError("amount is required for buy order")

        payload["amount"] = amount
    elif side == "SELL":
        if not quantity:
            raise ValueError("quantity is required for sell order")

        payload["qty"] = quantity

    # if limit_price:
    #     payload["limit_price"] = limit_price

    data = get_response(action="/v2.1/order", payload=payload)
    return OrderResponse(**data)


def buy_ticker(ticker, amount_krw) -> OrderResponse:
    return _order(ticker, "BUY", amount=amount_krw)


def sell_ticker(ticker, quantity, limit_price) -> OrderResponse:
    return _order(ticker, "SELL", quantity=quantity, limit_price=limit_price)


def get_completed_orders(ticker, size=10, from_at: datetime = None, to_at: datetime = None):
    to_at = to_at or timezone.now()
    to_ts = int(to_at.timestamp() * 1000)
    from_at = from_at or to_at - timedelta(days=1)
    from_ts = int(from_at.timestamp() * 1000)

    return get_response(
        action=f"/v2.1/order/completed_orders",
        payload={
            "access_token": ACCESS_TOKEN,
            "quote_currency": "KRW",
            "target_currency": ticker,
            "size": size,
            "from_ts": from_ts,
            "to_ts": to_ts,
        },
    )


def get_order_detail(order_id, target_currency):
    return get_response(
        action=f"/v2.1/order/detail",
        payload={
            "access_token": ACCESS_TOKEN,
            "order_id": order_id,
            "quote_currency": "KRW",
            "target_currency": target_currency,
        },
    )


def get_markets():
    data = get_response(action="/public/v2/markets/KRW", method="get", public=True)
    return {market["target_currency"]: market for market in data["markets"]}


def get_candles(ticker: str, interval: str, size: int = 200):
    """코인원 캔들차트 데이터를 조회합니다.
    :param ticker: 조회할 종목 (예: BTC, SOL)
    :param interval: 캔들 간격 (예: 1m, 5m, 1h 등)
    :param size: 조회할 캔들 수 (최소 1~최대 500) 미입력시 default: 200
    :return: 캔들 데이터 JSON 응답
    """
    return get_response(
        action=f"/public/v2/chart/KRW/{ticker}",
        method="get",
        query={"interval": interval, "size": size},
        public=True,
    )


def get_orderbook(ticker: str, size: int = 15):
    """코인원 오더북 데이터를 조회합니다.
    :param ticker: 조회할 종목 (예: BTC, SOL)
    :param size: 오더북 개수 (예: 5, 10, 15, 16만 허용. 기본 15)
    :return: 오더북 데이터 JSON 응답
    """
    return get_response(
        action=f"/public/v2/orderbook/KRW/{ticker}",
        method="get",
        query={"size": size},
        public=True,
    )


def get_trades(ticker: str, size: int = 200):
    """코인원 최근 체결 데이터를 조회합니다.
    :param ticker: 조회할 종목 (예: BTC, SOL)
    :param size: 최근 체결 개수 (예: 10, 50, 100, 150, 200만 허용. 기본 200)
    :return: 최근 체결 데이터 JSON 응답
    """
    return get_response(
        action=f"/public/v2/trades/KRW/{ticker}",
        method="get",
        public=True,
        query={"size": size},
    )
