from django.db import models


class ExchangeChoices(models.TextChoices):
    COINONE = "coinone"
    UPBIT = "upbit"
