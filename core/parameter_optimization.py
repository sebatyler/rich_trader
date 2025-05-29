"""
파라미터 최적화 모듈

이 모듈은 최근 거래 데이터와 손익 결과를 기반으로 매매 알고리즘의 파라미터를 최적화합니다.
Gemini 2.5 Pro API를 통해 최적화된 파라미터를 요청하고 수신하며,
수신된 파라미터를 알고리즘에 적용하는 기능을 제공합니다.
"""

import decimal
import json
from datetime import date
from datetime import datetime

from pydantic import BaseModel
from pydantic import Field

from core.llm import invoke_llm


class OptimizedParameters(BaseModel):
    reasoning: str = Field(..., description="최적화된 파라미터에 대한 근거 설명")
    rsi_period: int
    bollinger_period: int
    bollinger_std: float
    buy_rsi_threshold: float
    sell_rsi_threshold: float
    buy_pressure_threshold: float
    sell_pressure_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    buy_profit_rate: float
    sell_profit_rate: float
    max_krw_buy_ratio: float


def json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def create_optimization_payload(trade_data, portfolio, current_parameters=None):
    """
    최근 거래 데이터와 포트폴리오 상태를 바탕으로 파라미터 최적화 요청 payload를 생성합니다.

    :param trade_data: 최근 거래 데이터 (예: 리스트 형태)
    :param portfolio: 최근 포트폴리오 상태 (dict)
    :param current_parameters: 현재 파라미터 (dict)
    :return: 최적화 요청 payload (dict)
    """
    if current_parameters is None:
        current_parameters = {
            "rsi_period": 14,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "buy_rsi_threshold": 30.0,
            "sell_rsi_threshold": 70.0,
            "buy_pressure_threshold": 0.6,
            "sell_pressure_threshold": 0.4,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "buy_profit_rate": -5.0,
            "sell_profit_rate": 5.0,
            "max_krw_buy_ratio": 0.1,
        }
    payload = {
        "trade_data": trade_data,
        "portfolio": portfolio,
        "current_parameters": current_parameters,
    }
    return payload


def request_optimized_parameters(payload):
    """
    Gemini API를 직접 호출하는 대신, LLM 함수를 이용하여 최적화된 파라미터를 생성합니다.

    :param payload: 최적화 요청 payload (dict)
    :return: 최적화된 파라미터 (OptimizedParameters) 또는 None if 실패
    """
    prompt = (
        "아래는 최근 거래 데이터와 포트폴리오 상태입니다. 이 정보를 바탕으로 트레이딩 알고리즘의 최적화된 파라미터를 제안해 주세요. "
        "최적화 파라미터는 'rsi_period', 'bollinger_period', 'bollinger_std', 'buy_rsi_threshold', 'sell_rsi_threshold', "
        "'buy_pressure_threshold', 'sell_pressure_threshold', 'stop_loss_pct', 'take_profit_pct', "
        "'buy_profit_rate', 'sell_profit_rate', 'max_krw_buy_ratio'를 반드시 포함해야 합니다.\n"
        "조건: buy_profit_rate는 반드시 음수(매수 허용 최대 손실률), sell_profit_rate는 반드시 양수(매도 허용 최소 수익률)여야 합니다.\n"
        "- stop_loss_pct, take_profit_pct는 소수(예: 0.02 = 2%)로 입력해야 합니다.\n"
        "- buy_profit_rate, sell_profit_rate는 정수 또는 실수(예: 5 = 5%)로 입력해야 합니다.\n"
        "데이터:\n"
    )
    message = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default).replace("{", "{{").replace("}", "}}")
    )
    optimized_params = invoke_llm(
        prompt, message, with_fallback=True, model=OptimizedParameters, structured_output=True
    )
    return optimized_params
