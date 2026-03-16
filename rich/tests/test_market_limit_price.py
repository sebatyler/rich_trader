import pytest

from core.coinone import OrderResponse
from rich.service import Recommendation
from rich.service import _compute_market_order_limit_price
from rich.service import execute_trade


def test_compute_market_order_limit_price_skips_tight_spreads():
    assert (
        _compute_market_order_limit_price(
            action="BUY",
            snapshot={"best_ask": 100.0, "best_bid": 99.95, "spread_pct": 0.05},
            market={"price_unit": "0.1", "min_price": "0.1", "max_price": "100000"},
        )
        is None
    )


def test_compute_market_order_limit_price_rounds_buy_and_sell_guards():
    market = {"price_unit": "0.1", "min_price": "0.1", "max_price": "100000"}
    snapshot = {
        "best_ask": 100.0,
        "best_bid": 99.7,
        "spread_pct": ((100.0 / 99.7) - 1) * 100,
    }

    assert _compute_market_order_limit_price("BUY", snapshot, market) == 100.2
    assert _compute_market_order_limit_price("SELL", snapshot, market) == 99.5


@pytest.mark.django_db
def test_execute_trade_applies_market_limit_guard_to_buy_orders(
    mocker,
    settings,
    user_factory,
):
    settings.DEBUG = False
    user = user_factory()
    recommendation = Recommendation(
        action="BUY",
        symbol="BTC",
        amount=10_000,
        quantity=None,
        limit_price=None,
        reason="test",
    )
    crypto_data = {
        "input_data": {
            "current_price": 100.0,
            "best_ask": 100.0,
            "best_bid": 99.7,
            "spread_pct": ((100.0 / 99.7) - 1) * 100,
        }
    }
    market = {"price_unit": "0.1", "min_price": "0.1", "max_price": "100000"}
    buy_ticker = mocker.patch(
        "rich.service.coinone.buy_ticker",
        return_value=OrderResponse(result="success", error_code="0", order_id="order-1"),
    )
    mocker.patch(
        "rich.service.coinone.get_order_detail",
        return_value={
            "order": {
                "order_id": "order-1",
                "type": "MARKET",
                "side": "BUY",
                "status": "done",
                "fee": "0",
            }
        },
    )
    process_trade = mocker.patch("rich.service.process_trade", return_value=({}, None))

    execute_trade(
        user=user,
        recommendation=recommendation,
        crypto_data=crypto_data,
        chat_id="chat-id",
        market=market,
    )

    buy_ticker.assert_called_once_with("BTC", 10_000, limit_price=100.2)
    assert process_trade.call_args.kwargs["limit_price"] == 100.2
