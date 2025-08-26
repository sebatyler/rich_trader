from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from django.http import HttpResponse
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.utils import timezone
from django.views import View
from django.views.generic import TemplateView

from core import upbit
from core.choices import ExchangeChoices
from core.utils import format_quantity
from trading.forms import TradingConfigForm
from trading.models import Portfolio
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
        now = timezone.localtime()
        today = now.date()

        # 최신 포트폴리오 데이터 조회
        portfolio_queryset = Portfolio.objects.filter(exchange=ExchangeChoices.UPBIT).order_by("-id")
        recent_portfolios = list(portfolio_queryset[:2])

        # 현재 포트폴리오 데이터 (가장 최신)
        current_portfolio = recent_portfolios[0]
        current_data = current_portfolio.export()

        # 비교용 포트폴리오들 수집
        comparison_portfolios = [
            # 직전 포트폴리오
            recent_portfolios[1],
            # 오늘 첫 포트폴리오
            portfolio_queryset.filter(created__date=today).order_by("created").first(),
            # 24시간 전 포트폴리오
            portfolio_queryset.filter(created__lt=now - timedelta(hours=24)).order_by("-created").first(),
        ]

        # 코인별 가격 변화율 계산
        price_changes = defaultdict(list)
        price_change_timestamps = defaultdict(list)

        for historical_portfolio in comparison_portfolios:
            if not historical_portfolio:
                continue

            historical_data = historical_portfolio.export()

            for current_balance in current_data["balances"]:
                for historical_balance in historical_data["balances"]:
                    if current_balance["symbol"] == historical_balance["symbol"]:
                        historical_price = Decimal(historical_balance["current_price"])
                        current_price = Decimal(current_balance["current_price"])

                        # 가격 변화율 계산 (백분율)
                        price_change_percentage = (current_price / historical_price - 1) * 100

                        price_changes[current_balance["symbol"]].append(price_change_percentage)
                        price_change_timestamps[current_balance["symbol"]].append(
                            historical_portfolio.created.astimezone()
                        )
                        break

        # 결과 데이터에 변화율 정보 추가
        current_data.update(
            {
                "price_changes": price_changes,
                "price_change_timestamps": price_change_timestamps,
            }
        )

        return current_data


class UpbitBalanceView(UpbitMixin, View):
    """업비트 잔고를 JSON으로 반환하는 뷰"""

    def get(self, request, *args, **kwargs):
        return JsonResponse(self.get_upbit_data())


class UpbitBalanceTemplateView(UpbitMixin, TemplateView):
    template_name = "upbit_balance.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        data = self.get_upbit_data()
        context.update(data)
        return context
