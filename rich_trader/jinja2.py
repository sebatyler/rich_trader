from decimal import Decimal

from jinja2 import Environment

from django.conf import settings
from django.templatetags.static import static
from django.urls import reverse


def environment(**options):
    env = Environment(**options)
    env.globals.update(
        {
            "static": static,
            "url": reverse,
            "firebase_config": settings.FIREBASE_CONFIG,
            "Decimal": Decimal,
        }
    )
    return env
