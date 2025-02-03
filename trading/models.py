from decimal import Decimal

import pandas as pd
from model_utils.models import TimeStampedModel
from simple_history.models import HistoricalRecords

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from accounts.models import User
from core.choices import ExchangeChoices
from core.models import choice_field


class TradingConfig(TimeStampedModel):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    coinone_access_key = models.CharField(max_length=255)
    coinone_secret_key = models.CharField(max_length=255)
    telegram_chat_id = models.CharField(max_length=255)
    is_active = models.BooleanField(
        verbose_name="자동 매매 활성화",
        help_text="체크하면 자동 매매가 활성화됩니다",
        default=True,
    )
    target_coins = models.JSONField(
        help_text="List of target coins",
        default=list,
    )
    min_trade_amount = models.PositiveIntegerField(
        verbose_name="최소 거래금액",
        help_text="거래당 최소 금액 (원)",
        default=5_000,
    )
    step_amount = models.PositiveIntegerField(
        verbose_name="거래금액 단위",
        help_text="거래금액의 증가 단위 (원)",
        default=5_000,
    )
    min_amount = models.PositiveIntegerField(
        verbose_name="최소 매수금액",
        help_text="한 번에 매수할 최소 금액 (원)",
        default=5_000,
    )
    max_amount = models.PositiveIntegerField(
        verbose_name="최대 매수금액",
        help_text="한 번에 매수할 최대 금액 (원)",
        default=30_000,
    )
    min_coins = models.SmallIntegerField(
        verbose_name="최소 코인 개수",
        help_text="한 번에 추천할 최소 코인 개수 (0은 거래 추천이 없을 수 있음)",
        default=1,
    )
    max_coins = models.PositiveSmallIntegerField(
        verbose_name="최대 코인 개수",
        help_text="한 번에 추천할 최대 코인 개수",
        default=2,
    )

    history = HistoricalRecords()

    class Meta:
        verbose_name = "Trading Configuration"
        verbose_name_plural = "Trading Configurations"

    def __str__(self):
        return f"{self.user.username}'s Trading Config"

    def clean(self):
        if self.min_trade_amount <= 0:
            raise ValidationError("Minimum trade amount must be positive")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class Trading(TimeStampedModel):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    order_id = models.CharField(max_length=255)
    coin = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True, help_text="주문 금액 (KRW)")
    quantity = models.DecimalField(max_digits=17, decimal_places=8, null=True, blank=True, help_text="주문 수량 (코인)")
    limit_price = models.DecimalField(
        max_digits=20, decimal_places=0, null=True, blank=True, help_text="주문 제한가 (KRW)"
    )
    reason = models.TextField(null=True, blank=True, help_text="주문 사유")

    type = models.CharField(max_length=20, help_text="주문 유형 (예: MARKET)")
    side = models.CharField(max_length=10, help_text="BUY/SELL")
    status = models.CharField(max_length=50)
    fee = models.DecimalField(max_digits=20, decimal_places=0, help_text="거래 수수료 (KRW)")
    price = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True, help_text="주문 가격 (KRW)")
    fee_rate = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True, help_text="수수료율 (%)")
    average_executed_price = models.DecimalField(
        max_digits=20, decimal_places=0, null=True, blank=True, help_text="평균 체결 가격 (KRW)"
    )
    average_fee_rate = models.DecimalField(
        max_digits=5, decimal_places=4, null=True, blank=True, help_text="평균 수수료율 (%)"
    )
    original_qty = models.DecimalField(
        max_digits=17, decimal_places=8, null=True, blank=True, help_text="최초 주문 수량 (코인)"
    )
    executed_qty = models.DecimalField(
        max_digits=17, decimal_places=8, null=True, blank=True, help_text="체결된 수량 (코인)"
    )
    canceled_qty = models.DecimalField(
        max_digits=17, decimal_places=8, null=True, blank=True, help_text="취소된 수량 (코인)"
    )
    traded_amount = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True, help_text="체결된 총액")
    original_amount = models.DecimalField(max_digits=20, decimal_places=0, null=True, blank=True, help_text="주문 총액")
    canceled_amount = models.DecimalField(
        max_digits=20, decimal_places=0, blank=True, null=True, help_text="주문 취소 총액"
    )

    order_detail = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.user.username}'s {self.side} {self.coin} order ({self.order_id})"

    def save(self, *args, **kwargs):
        is_adding = self._state.adding

        if is_adding:
            if self.order_detail and (data := self.order_detail.get("order")):
                for field in (
                    "fee_rate",
                    "average_executed_price",
                    "average_fee_rate",
                    "limit_price",
                    "original_qty",
                    "executed_qty",
                    "canceled_qty",
                    "traded_amount",
                    "original_amount",
                    "canceled_amount",
                ):
                    if getattr(self, field) is None:
                        setattr(self, field, data.get(field))

            for field in self._meta.fields:
                if isinstance(field, models.DecimalField):
                    value = getattr(self, field.name)
                    if value is not None and not isinstance(value, Decimal):
                        setattr(self, field.name, Decimal(str(value)))

        super().save(*args, **kwargs)

    @classmethod
    def get_recent_trades_csv(cls, user: User, limit: int = 20):
        return pd.DataFrame(
            cls.objects.filter(user=user, executed_qty__gt=0)
            .order_by("-id")[:limit]
            .values("coin", "side", "executed_qty", "average_executed_price", "fee", "created")
        ).to_csv(index=False)


class Portfolio(TimeStampedModel):
    exchange = choice_field(ExchangeChoices, default=ExchangeChoices.COINONE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    balances = models.JSONField(default=list)
    total_portfolio_value = models.PositiveBigIntegerField(help_text="총 포트폴리오 가치 (KRW)")
    krw_balance = models.PositiveBigIntegerField(help_text="KRW 잔액 (KRW)")
    total_coin_value = models.PositiveBigIntegerField(help_text="총 코인 가치 (KRW)")

    @property
    def krw_weight(self):
        return self.krw_balance / self.total_portfolio_value * 100

    def export(self):
        return {
            "balances": self.balances,
            "total_coin_value": self.total_coin_value,
            "krw_value": self.krw_balance,
            "krw_weight": self.krw_weight,
            "total_value": self.total_portfolio_value,
        }
