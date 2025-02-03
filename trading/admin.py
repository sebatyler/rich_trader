from django_json_widget.widgets import JSONEditorWidget
from simple_history.admin import SimpleHistoryAdmin

from django.contrib import admin
from django.db import models

from core.admin import ModelAdmin

from .models import Portfolio
from .models import Trading
from .models import TradingConfig


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
    formfield_overrides = {
        models.JSONField: {"widget": JSONEditorWidget},
    }


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
    formfield_overrides = {
        models.JSONField: {"widget": JSONEditorWidget},
    }
    readonly_fields = ("krw_weight", "created")

    def krw_weight(self, obj):
        return f"{obj.krw_weight:.2f}%"
