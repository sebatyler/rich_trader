from django_json_widget.widgets import JSONEditorWidget
from simple_history.admin import SimpleHistoryAdmin

from django.contrib import admin
from django.db import models

from core.admin import ModelAdmin

from .models import AlgorithmParameter
from .models import AutoTrading
from .models import Portfolio
from .models import Trading
from .models import TradingConfig
from .models import UpbitTrading

formfield_overrides = {
    models.JSONField: {"widget": JSONEditorWidget},
}


@admin.register(TradingConfig)
class TradingConfigAdmin(SimpleHistoryAdmin, ModelAdmin):
    list_display = (
        "id",
        "user",
        "is_active",
        "target_coins",
        "min_trade_amount",
        "step_amount",
        "min_coins",
        "max_coins",
        "created",
        "modified",
    )
    list_filter = ("is_active",)
    list_select_related = ("user",)
    search_fields = ("user__username", "user__email", "target_coins")
    list_display_links = ("id", "user")


@admin.register(Trading)
class TradingAdmin(ModelAdmin):
    list_display = (
        "id",
        "user",
        "coin",
        "type",
        "side",
        "price",
        "amount",
        "quantity",
        "limit_price",
        "executed_qty",
        "status",
        "created",
    )
    list_filter = ("user", "coin", "type", "side", "status")
    list_select_related = ("user",)
    search_fields = ("user__username", "user__email", "coin")
    list_display_links = ("id", "user")
    formfield_overrides = formfield_overrides


@admin.register(Portfolio)
class PortfolioAdmin(ModelAdmin):
    list_display = (
        "id",
        "exchange",
        "user",
        "total_portfolio_value",
        "krw_balance",
        "total_coin_value",
        "krw_weight",
        "created",
    )
    list_select_related = ("user",)
    list_filter = ("exchange", "user")
    search_fields = ("user__username", "user__email")
    formfield_overrides = formfield_overrides
    readonly_fields = ("krw_weight", "created")

    def krw_weight(self, obj):
        return f"{obj.krw_weight:.2f}%"


@admin.register(UpbitTrading)
class UpbitTradingAdmin(ModelAdmin):
    list_display = (
        "id",
        "coin",
        "amount",
        "is_dca",
        "paid_fee",
        "executed_volume",
        "average_price",
        "created",
    )
    list_filter = ("coin", "is_dca")
    formfield_overrides = formfield_overrides


@admin.register(AlgorithmParameter)
class AlgorithmParameterAdmin(ModelAdmin):
    list_display = (
        "id",
        "user",
        "rsi_period",
        "bollinger_period",
        "bollinger_std",
        "buy_rsi_threshold",
        "sell_rsi_threshold",
        "created",
    )
    list_filter = ("user",)
    search_fields = ("user__username",)
    list_display_links = ("id", "user")


@admin.register(AutoTrading)
class AutoTradingAdmin(ModelAdmin):
    list_display = (
        "id",
        "created",
        "is_processing",
        "trading",
        "signal",
        "stop_loss_signal",
        "rsi",
        "bollinger_upper",
        "bollinger_lower",
        "current_price",
        "btc_available",
        "krw_available",
        "finished_at",
    )
    list_filter = (
        "is_processing",
        ("trading", admin.EmptyFieldListFilter),
        "signal",
        "stop_loss_signal",
    )
    list_select_related = ("trading",)
    search_fields = ("id",)
    list_display_links = ("id",)
    readonly_fields = ("created",)
