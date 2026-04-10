from decimal import Decimal
from io import StringIO

import pandas as pd
import pytest

from trading.models import Trading
from trading.models import BybitMechanicalParameter
from trading.models import BybitMechanicalTrade


@pytest.mark.django_db
def test_get_recent_trades_csv_adds_current_price_and_price_diff_cols(
    user_factory, trading_factory
):
    user = user_factory()
    trading_factory(
        user=user,
        coin="BTC",
        side="SELL",
        executed_qty=Decimal("0.01"),
        average_executed_price=Decimal("120000000"),
    )

    csv_data = Trading.get_recent_trades_csv(
        user=user,
        current_prices={"BTC": 90000000},
    )
    df = pd.read_csv(StringIO(csv_data))

    assert "current_price" in df.columns
    assert "price_diff_pct" in df.columns
    btc = df.loc[df["coin"] == "BTC"].iloc[0]
    assert btc["current_price"] == pytest.approx(90000000)
    assert btc["price_diff_pct"] == pytest.approx(-25.0)


@pytest.mark.django_db
def test_get_recent_trades_csv_handles_zero_average_price(
    user_factory, trading_factory
):
    user = user_factory()
    trading_factory(
        user=user,
        coin="XRP",
        executed_qty=Decimal("5"),
        average_executed_price=Decimal("0"),
    )

    csv_data = Trading.get_recent_trades_csv(
        user=user,
        current_prices={"XRP": 1000},
    )
    df = pd.read_csv(StringIO(csv_data))

    xrp = df.loc[df["coin"] == "XRP"].iloc[0]
    assert xrp["current_price"] == pytest.approx(1000)
    assert pd.isna(xrp["price_diff_pct"])


@pytest.mark.django_db
def test_bybit_mechanical_models_do_not_require_user():
    params = BybitMechanicalParameter.objects.create()
    trade = BybitMechanicalTrade.objects.create(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=Decimal("100000"),
        position_size_usd=Decimal("100"),
        leverage=2,
    )

    assert "user" not in {field.name for field in BybitMechanicalParameter._meta.fields}
    assert "user" not in {field.name for field in BybitMechanicalTrade._meta.fields}
    assert str(params).startswith("BybitParam @")
    assert trade.symbol == "BTCUSDT"
