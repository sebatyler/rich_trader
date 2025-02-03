from decimal import Decimal

from django.http import HttpResponse
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views import View
from django.views.generic import TemplateView

from core import upbit
from core.utils import format_quantity
from trading.forms import TradingConfigForm
from trading.models import TradingConfig


class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        if self.request.user.is_authenticated:
            trading_config = TradingConfig.objects.filter(user=self.request.user).first()
            if not trading_config:
                trading_config = TradingConfig.objects.create(
                    user=self.request.user,
                    is_active=False,
                )

            form = TradingConfigForm(instance=trading_config)
            context["form"] = form

        return context

    def post(self, request, *args, **kwargs):
        form = TradingConfigForm(request.POST, instance=request.user.tradingconfig)
        if form.is_valid():
            form.save()
            if request.headers.get("HX-Request"):
                html = render_to_string(
                    "components/alert.html",
                    {"message": "트레이딩 설정이 업데이트 되었습니다!"},
                    request,
                )
                return HttpResponse(html)

        context = self.get_context_data(**kwargs)
        context["form"] = form
        return self.render_to_response(context)


class UpbitMixin(View):
    def dispatch(self, request, *args, **kwargs):
        # 슈퍼유저 또는 특정 쿼리 파라미터가 있는 경우만 접근 가능
        if not request.user.is_superuser and request.GET.get("from") != "chatgpt_seba":
            return JsonResponse({"error": "Permission denied"}, status=403)

        return super().dispatch(request, *args, **kwargs)

    def get_upbit_data(self):
        return upbit.get_balance_data()


class UpbitBalanceView(UpbitMixin, View):
    """업비트 잔고를 JSON으로 반환하는 뷰"""

    def get(self, request, *args, **kwargs):
        return JsonResponse(self.get_upbit_data())


class UpbitBalanceTemplateView(UpbitMixin, TemplateView):
    template_name = "upbit_balance.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        data = self.get_upbit_data()

        # float 타입으로 변환
        for balance_dict in data["balances"]:
            update_dict = {}
            for key, value in balance_dict.items():
                if key in ("quantity", "avg_buy_price", "current_price"):
                    val = balance_dict[key] = Decimal(value)
                    formatted = format_quantity(val)
                    if key.endswith("_price") and val >= 100:
                        formatted = formatted.split(".")[0]
                    update_dict[f"{key}_display"] = formatted
                elif isinstance(value, str) and value[0].isdigit():
                    balance_dict[key] = float(value)
            balance_dict.update(update_dict)

        context.update(data)
        return context
