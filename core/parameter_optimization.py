"""
파라미터 최적화 모듈

이 모듈은 최근 거래 데이터와 손익 결과를 기반으로 매매 알고리즘의 파라미터를 최적화합니다.
Gemini 2.5 Pro API를 통해 최적화된 파라미터를 요청하고 수신하며,
수신된 파라미터를 알고리즘에 적용하는 기능을 제공합니다.
"""

import json

from pydantic import BaseModel


class OptimizedParameters(BaseModel):
    rsi_period: int
    bollinger_period: int
    bollinger_std: int


def create_optimization_payload(trade_data, profit_loss):
    """
    최근 거래 데이터와 총 손익을 바탕으로 파라미터 최적화 요청 payload를 생성합니다.

    :param trade_data: 최근 거래 데이터 (예: 리스트 형태)
    :param profit_loss: 총 손익 (float)
    :return: 최적화 요청 payload (dict)
    """
    payload = {
        "trade_data": trade_data,
        "profit_loss": profit_loss,
        "current_parameters": {"rsi_period": 14, "bollinger_period": 20, "bollinger_std": 2},
    }
    return payload


def request_optimized_parameters(payload):
    """
    Gemini API를 직접 호출하는 대신, LLM 함수를 이용하여 최적화된 파라미터를 생성합니다.

    :param payload: 최적화 요청 payload (dict)
    :return: 최적화된 파라미터 (OptimizedParameters) 또는 None if 실패
    """
    from llm import invoke_llm

    prompt = (
        "Based on the data provided below, please propose optimized parameters for the trading algorithm. "
        "The optimized parameters should include 'rsi_period', 'bollinger_period', and 'bollinger_std'.\n"
        "Data: \n"
    )
    prompt += json.dumps(payload, ensure_ascii=False, indent=2)
    optimized_params = invoke_llm(prompt, with_fallback=True, model=OptimizedParameters)
    return optimized_params


def apply_optimized_parameters(optimized_params):
    """
    최적화된 파라미터를 현재 알고리즘 설정에 적용합니다.

    :param optimized_params: 최적화된 파라미터 (dict)
    :return: None
    """
    if not optimized_params:
        print("적용할 최적화 파라미터가 없습니다.")
        return

    # 실제 환경에서는 데이터베이스 업데이트 또는 전역 변수 업데이트 등을 수행할 수 있습니다.
    # 아래는 예시로 최적화된 파라미터를 출력합니다.
    print("최적화된 파라미터가 적용되었습니다:", optimized_params)
