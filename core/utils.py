from decimal import Decimal


def format_currency(value):
    """통화 값을 읽기 쉬운 형식으로 변환합니다."""
    if value > 1_000_000_000_000:
        value = f"{value/1_000_000_000_000:,.2f}T"
    elif value > 1_000_000_000:
        value = f"{value/1_000_000_000:,.2f}B"
    elif value > 1_000_000:
        value = f"{value/1_000_000:,.2f}M"
    elif value > 1_000:
        value = f"{value/1_000:,.2f}K"

    return value


def dict_pick(data: dict, *keys: list[str]) -> dict:
    """딕셔너리에서 특정 키만 선택하여 반환합니다."""
    return {k: data[k] for k in keys if k in data}


def dict_at(data: dict, *keys: list[str]) -> list:
    """딕셔너리에서 특정 키만 선택하여 리스트로 반환합니다."""
    return [data.get(k) for k in keys]


def dict_omit(data: dict, *keys: list[str]) -> dict:
    """딕셔너리에서 특정 키를 제거하여 반환합니다."""
    return {k: v for k, v in data.items() if k not in keys}


def format_quantity(quantity: Decimal) -> str:
    """수량을 읽기 쉬운 형식으로 변환합니다."""
    display = f"{quantity:,.8f}"

    if "." in display:
        display = display.rstrip("0").rstrip(".")

    return display


def get_max_length(input_list):
    return max(map(len, filter(None, input_list)), default=None)
