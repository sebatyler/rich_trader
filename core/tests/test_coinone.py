from core import coinone


def test_buy_ticker_passes_limit_price_to_order_payload(mocker):
    mock_response = mocker.patch(
        "core.coinone.get_response",
        return_value={"result": "success", "error_code": "0", "order_id": "order-1"},
    )
    coinone.ACCESS_TOKEN = "access-token"

    coinone.buy_ticker("BTC", 10_000, limit_price=101.5)

    assert mock_response.call_args.kwargs["payload"]["limit_price"] == 101.5

