"""rich_trader URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include
from django.urls import path

from accounts.views import admin_login
from rich.views import IndexView
from rich.views import UpbitBalanceTemplateView
from rich.views import UpbitBalanceView

urlpatterns = [
    path("_a/login/", admin_login, name="admin_login"),
    path("_a/", admin.site.urls),
    path("", IndexView.as_view(), name="index"),
    path("upbit/balance/", UpbitBalanceTemplateView.as_view(), name="upbit_balance"),
    path("accounts/", include("accounts.urls")),
    path("api/upbit/balances/", UpbitBalanceView.as_view(), name="api_upbit_balances"),
]
