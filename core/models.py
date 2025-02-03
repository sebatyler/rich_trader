from django.db import models

from core.utils import get_max_length


def choice_field(choice_class, **kwargs):
    if issubclass(choice_class, models.TextChoices):
        field_class = models.CharField
        kwargs["max_length"] = kwargs.get("max_length", get_max_length(choice_class.values))
    elif issubclass(choice_class, models.IntegerChoices):
        field_class = models.PositiveSmallIntegerField
    else:
        raise ValueError(f"wrong choice_class: {choice_class}")

    return field_class(choices=choice_class.choices, **kwargs)
