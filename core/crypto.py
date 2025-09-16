import os

import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from core.llm import invoke_gemini_search_json
import yfinance as yf

from django.conf import settings


def fetch_coinmarketcap_data(path, params=None):
    """CoinMarketCap API를 호출하여 데이터를 가져옵니다."""
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY"),
    }
    url = "https://pro-api.coinmarketcap.com" + path

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.json()["data"]


def get_latest_listings(min_market_cap=1_000, limit=1000):
    """최신 암호화폐 목록을 가져옵니다."""
    params = {
        "convert": "USD",
        "cryptocurrency_type": "coins",
        "market_cap_min": min_market_cap,
        "limit": limit,
    }
    return fetch_coinmarketcap_data("/v1/cryptocurrency/listings/latest", params)


def get_quotes(symbol):
    """특정 암호화폐의 최신 시세 정보를 가져옵니다."""
    return fetch_coinmarketcap_data(
        "/v2/cryptocurrency/quotes/latest",
        params={"symbol": symbol, "convert": "KRW"},
    )[
        symbol
    ][0]


def get_historical_data(fsym, tsym, limit):
    """특정 암호화폐의 과거 데이터를 가져옵니다."""
    url = "https://min-api.cryptocompare.com/data/histoday"
    parameters = {"fsym": fsym, "tsym": tsym, "limit": limit}
    response = requests.get(url, params=parameters)
    data = response.json()
    return data["Data"]


def get_network_stats():
    """블록체인 네트워크 통계를 가져옵니다."""
    url = "https://api.blockchain.info/stats"
    response = requests.get(url)
    return response.json()


def fetch_news(from_date, query, page_size=10):
    """뉴스 API를 통해 암호화폐 관련 뉴스를 가져옵니다."""
    url = "https://newsapi.org/v2/everything"
    parameters = {"q": query, "from": from_date, "pageSize": page_size}
    response = requests.get(
        url,
        params=parameters,
        headers={"X-Api-Key": os.getenv("NEWS_API_KEY")},
    )
    data = response.json()
    return data["articles"]


def _normalize_url(url: str) -> str:
    """Remove common tracking parameters and normalize URL for deduplication."""
    try:
        parts = urlparse(url)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
        clean = parts._replace(query=urlencode(query, doseq=True))
        # Remove trailing '?' if query is empty
        normalized = urlunparse(clean).rstrip("?")
        return normalized
    except Exception:
        return url


def _to_newsapi_article(item: dict) -> dict:
    """Convert Gemini item into a NewsAPI-like article schema."""
    published = item.get("published_at_utc") or item.get("published_at")
    # Ensure ISO 8601 in UTC with 'Z'
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00")) if isinstance(published, str) else None
        published_at = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None
    except Exception:
        published_at = published

    return {
        "source": {"name": item.get("source") or "web"},
        "author": None,
        "title": item.get("title"),
        "description": item.get("summary") or item.get("title"),
        "url": _normalize_url(item.get("url", "")),
        "urlToImage": None,
        "publishedAt": published_at,
        "content": item.get("summary") or item.get("title"),
    }


def fetch_news_with_gemini_gap(from_date: str, symbol: str, page_size: int = 10, gap_threshold_minutes: int = 15,
                               max_window_hours: int = 12) -> list[dict]:
    """Fetch news via API and backfill the freshness gap using Gemini web search between latest API ts and now.

    Returns a list of articles matching NewsAPI schema.
    """
    api_articles = []
    try:
        api_articles = fetch_news(from_date, symbol, page_size)
    except Exception:
        # On API failure, continue with empty list and rely on Gemini
        api_articles = []

    # Latest timestamp from API
    latest_api_ts = None
    for a in api_articles:
        ts = a.get("publishedAt")
        try:
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if not latest_api_ts or dt > latest_api_ts:
                    latest_api_ts = dt
        except Exception:
            continue

    now_utc = datetime.now(timezone.utc)

    # Decide whether to search the gap
    should_search = False
    start_utc = None
    if latest_api_ts:
        # Ensure timezone-aware UTC
        if latest_api_ts.tzinfo is None:
            latest_api_ts = latest_api_ts.replace(tzinfo=timezone.utc)
        else:
            latest_api_ts = latest_api_ts.astimezone(timezone.utc)
        start_utc = latest_api_ts + timedelta(seconds=1)
        should_search = (now_utc - start_utc) >= timedelta(minutes=gap_threshold_minutes)
    else:
        # No API results, search within a limited window ending now
        start_utc = now_utc - timedelta(hours=min(max_window_hours, 6))
        should_search = True

    gemini_items = []
    if should_search:
        start_iso = start_utc.isoformat().replace("+00:00", "Z")
        end_iso = now_utc.isoformat().replace("+00:00", "Z")

        # Known URLs to avoid duplicates
        known_urls = list({_normalize_url(a.get("url", "")) for a in api_articles if a.get("url")})

        aliases = {
            "BTC": ["BTC", "Bitcoin", "비트코인"],
            "ETH": ["ETH", "Ethereum", "이더리움"],
            "XRP": ["XRP", "리플"],
            "SOL": ["SOL", "Solana", "솔라나"],
        }
        target_aliases = aliases.get(symbol.upper(), [symbol])
        keywords = [
            "상장", "상장폐지", "입출금 중단", "해킹", "취약점", "ETF", "규제", "재판", "메인넷 업그레이드", "토큰 언락",
            "listing", "delisting", "outage", "halt", "exploit", "hack", "ETF", "SEC", "lawsuit", "mainnet upgrade", "token unlock",
        ]

        system_instruction = [
            "You are a high-precision crypto news retriever.",
            "Return only items within the given UTC time window in strict JSON.",
            "Discard unverifiable or undated items.",
        ]

        prompt = (
            "[Context]\n"
            f"- Asset aliases: {', '.join(target_aliases)}\n"
            f"- Time window (UTC): start={start_iso}, end={end_iso} (end exclusive)\n"
            f"- Known URLs: {known_urls}\n"
            f"- Extra keywords: {', '.join(keywords)}\n\n"
            "[Task]\n"
            "Search the web and return JSON with keys: window{start_utc,end_utc}, items[{title,url,source,language,published_at_utc,tickers,sentiment,impact_score,summary}]\n"
            "Exclude duplicates and items outside the window. JSON only."
        )

        try:
            result = invoke_gemini_search_json(prompt, system_instruction=system_instruction)
            items = (result or {}).get("items") or []
            # Map Gemini items to NewsAPI schema
            gemini_items = [_to_newsapi_article(x) for x in items if isinstance(x, dict)]
        except Exception:
            gemini_items = []

    # Merge & deduplicate
    seen = set()
    merged = []
    for item in (api_articles + gemini_items):
        url = _normalize_url(item.get("url", ""))
        title_key = (item.get("title") or "").strip().lower()
        key = (url, title_key)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    # Sort by publishedAt desc when available
    def _sort_key(a):
        ts = a.get("publishedAt")
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime(1970, 1, 1, tzinfo=timezone.utc)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    merged.sort(key=_sort_key, reverse=True)
    # Trim to requested page size
    return merged[:page_size]


def get_market_indices(start_date):
    """주요 시장 지표 데이터를 가져옵니다."""
    if not settings.DEBUG:
        yf.set_tz_cache_location("/tmp/yf")

    indices = ["CL=F", "^DJI", "^GSPC", "^IXIC"]
    indices_data = yf.download(indices, start=start_date)
    return indices_data["Close"].to_csv()
