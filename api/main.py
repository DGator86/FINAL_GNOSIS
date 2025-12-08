import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import your full Gnosis pipeline.
# If the import fails, we fall back to mock ideas.
try:
    from pipeline.full_pipeline import run_full_pipeline_for_symbol  # type: ignore
except Exception:  # pragma: no cover - defensive
    run_full_pipeline_for_symbol = None  # type: ignore

# Try to import the UnusualWhalesAdapter
try:
    from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
except Exception:
    UnusualWhalesAdapter = None  # makes API still run if adapter missing

logger = logging.getLogger(__name__)

app = FastAPI(title="Gnosis Backend API")

# Allow the Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load UW token
UW_TOKEN = (
    os.getenv("UNUSUAL_WHALES_API_TOKEN")
    or os.getenv("UNUSUAL_WHALES_API_KEY")
    or os.getenv("UNUSUAL_WHALES_TOKEN")
)

# Initialize adapter safely
uw_adapter: Optional[UnusualWhalesAdapter]
try:
    uw_adapter = UnusualWhalesAdapter(token=UW_TOKEN) if UnusualWhalesAdapter else None
    if getattr(uw_adapter, "use_stub", False):
        uw_adapter = None
except Exception:
    uw_adapter = None


# -----------------------------------------------------
# MODELS
# -----------------------------------------------------

class SymbolInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    marketCap: str
    beta: float
    description: str


class PricePoint(BaseModel):
    time: str
    price: float


class TradeIdea(BaseModel):
    id: str
    symbol: str
    title: str
    thesis: str
    direction: str
    confidence: float
    horizon: str


# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_direction(flow: dict) -> Optional[str]:
    candidates = [
        flow.get("direction"),
        flow.get("sentiment"),
        flow.get("side"),
        flow.get("order_side"),
        flow.get("option_type"),
        flow.get("type"),
    ]

    for cand in candidates:
        if not cand:
            continue
        c = str(cand).upper()
        if any(x in c for x in ["BUY", "CALL", "BULL"]):
            return "LONG"
        if any(x in c for x in ["SELL", "PUT", "BEAR"]):
            return "SHORT"
    return None


def _parse_expiration(flow: dict) -> Optional[datetime]:
    for k in ["expiration_date", "expiry", "expiration", "exp"]:
        raw = flow.get(k)
        if not raw:
            continue

        raw = str(raw).split("T")[0]
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                pass

    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None


def _determine_horizon(exp: Optional[datetime], flow: dict) -> str:
    dte = None

    for k in ["dte", "days_to_exp", "days_to_expiration"]:
        if flow.get(k) is not None:
            try:
                dte = int(float(flow[k]))
                break
            except Exception:
                pass

    if exp:
        try:
            dte = (exp.date() - datetime.utcnow().date()).days
        except Exception:
            pass

    if dte is None:
        return "SWING"
    if dte <= 1:
        return "INTRADAY"
    if dte <= 10:
        return "SWING"
    return "POSITION"


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Safely get attribute or dict key from an object using multiple candidate names."""
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
    return default


def _normalize_direction(raw: Optional[str]) -> str:
    if not raw:
        return "LONG"
    text = str(raw).upper()
    if any(tok in text for tok in ["SHORT", "PUT", "BEAR", "DOWN"]):
        return "SHORT"
    return "LONG"


def _normalize_horizon(raw: Optional[str]) -> str:
    if not raw:
        return "SWING"
    text = str(raw).upper()
    if "INTRA" in text or text in {"0DTE", "1D"}:
        return "INTRADAY"
    if any(tok in text for tok in ["SWING", "5D", "10D", "WEEK"]):
        return "SWING"
    if any(tok in text for tok in ["POS", "POSITION", "MONTH", "30D"]):
        return "POSITION"
    return "SWING"


def generate_trade_ideas_from_unusual_whales(symbol: str) -> List[TradeIdea]:
    symbol = symbol.upper()

    # If adapter missing OR token missing → fallback mock
    if not uw_adapter or not UW_TOKEN:
        logger.warning("UW adapter unavailable → using mock flow")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    # Call the user's real adapter
    try:
        flow_list = uw_adapter.get_unusual_activity(symbol)
    except Exception as e:
        logger.error(f"UW API error: {e}")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    ideas: List[TradeIdea] = []

    for idx, flow in enumerate(flow_list):
        direction = _parse_direction(flow)
        if not direction:
            continue

        exp = _parse_expiration(flow)
        horizon = _determine_horizon(exp, flow)

        premium = _coerce_float(
            flow.get("premium") or flow.get("notional") or flow.get("cost")
        )
        size = _coerce_float(flow.get("size") or flow.get("volume") or flow.get("contracts"))

        confidence = min(1.0, (premium / 250_000) + (size / 5_000))

        strike = flow.get("strike") or flow.get("strike_price") or "N/A"
        expiry_label = exp.strftime("%Y-%m-%d") if exp else flow.get("expiration_date")

        title = f"{symbol} {direction} @ {strike} exp {expiry_label}"
        thesis = (
            f"Unusual Whales flow detected for {symbol}. "
            f"Premium ~ ${premium:,.0f}. Size {size:,.0f}. "
            f"Expiry {expiry_label}. Use for confirmation."
        )

        ideas.append(
            TradeIdea(
                id=f"uw-{symbol}-{idx}",
                symbol=symbol,
                title=title,
                thesis=thesis,
                direction=direction,
                confidence=confidence,
                horizon=horizon,
            )
        )

    if not ideas:
        logger.info("UW flow returned zero usable trades; using mock ideas instead.")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    ideas.sort(key=lambda x: x.confidence, reverse=True)
    return ideas


def generate_trade_ideas_from_gnosis(symbol: str) -> List[TradeIdea]:
    """
    Call the real Gnosis engine pipeline (run_full_pipeline_for_symbol)
    and adapt whatever it returns into our TradeIdea schema.

    If anything fails, we fall back to the mock ideas for that symbol.
    """
    symbol = symbol.upper()

    # If the pipeline cannot be imported, use mocks.
    if run_full_pipeline_for_symbol is None:
        logger.warning("run_full_pipeline_for_symbol not importable; using mock trade ideas.")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    try:
        # You may need to tweak the arguments here to match your actual pipeline signature.
        # Common patterns from your repo are:
        #   run_full_pipeline_for_symbol(symbol)
        # or run_full_pipeline_for_symbol(symbol=symbol, mode="live")
        result = run_full_pipeline_for_symbol(symbol=symbol)  # type: ignore
    except Exception as exc:
        logger.error(f"Gnosis pipeline error for {symbol}: {exc}")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    # Extract raw proposed trades from the result, being defensive about structure.
    raw_trades: List[Any] = []

    if isinstance(result, dict):
        raw_trades = (
            result.get("proposed_trades")
            or result.get("trades")
            or result.get("trade_ideas")
            or []
        )
    else:
        raw_trades = (
            getattr(result, "proposed_trades", None)
            or getattr(result, "trades", None)
            or getattr(result, "trade_ideas", None)
            or []
        )

    if not isinstance(raw_trades, list):
        logger.warning("Gnosis pipeline returned non-list trades; using mock ideas.")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    ideas: List[TradeIdea] = []

    for idx, t in enumerate(raw_trades):
        if t is None:
            continue

        t_symbol = (_get(t, "symbol", "underlying", "ticker", default=symbol) or symbol).upper()
        direction = _normalize_direction(_get(t, "direction", "side", "bias", default="LONG"))
        thesis = _get(t, "rationale", "thesis", "notes", "comment",
                      default="No rationale provided by engine.")
        title = _get(t, "title", "label", "name",
                     default=f"{t_symbol} {direction.title()} idea")

        confidence = _coerce_float(_get(t, "confidence", "score", "probability", default=0.5), default=0.5)
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        horizon = _normalize_horizon(_get(t, "horizon", "timeframe", "window", default="SWING"))

        ideas.append(
            TradeIdea(
                id=f"gnosis-{t_symbol}-{idx}",
                symbol=t_symbol,
                title=title,
                thesis=thesis,
                direction=direction,
                confidence=confidence,
                horizon=horizon,
            )
        )

    if not ideas:
        logger.info("Gnosis pipeline returned zero usable trades; using mock ideas instead.")
        return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]

    # Sort by confidence descending so the strongest idea is first
    ideas.sort(key=lambda ti: ti.confidence, reverse=True)
    return ideas


# -----------------------------------------------------
# MOCK DATA (unchanged)
# -----------------------------------------------------
MOCK_SYMBOLS: Dict[str, SymbolInfo] = {
    "SPY": SymbolInfo(
        symbol="SPY",
        name="SPDR S&P 500 ETF",
        sector="Index / ETF",
        marketCap="$500B+",
        beta=1.0,
        description="Proxy for broad U.S. large-cap market."
    ),
    "QQQ": SymbolInfo(
        symbol="QQQ",
        name="Invesco QQQ",
        sector="Tech",
        marketCap="$250B+",
        beta=1.2,
        description="Tech-heavy ETF."
    ),
    "TSLA": SymbolInfo(
        symbol="TSLA",
        name="Tesla",
        sector="EV",
        marketCap="$500B+",
        beta=1.8,
        description="High beta single-name."
    ),
    "AAPL": SymbolInfo(
        symbol="AAPL",
        name="Apple",
        sector="Tech",
        marketCap="$3T+",
        beta=1.1,
        description="Mega-cap bellwether."
    ),
}

MOCK_PRICE_SERIES: Dict[str, List[PricePoint]] = {
    "SPY": [
        PricePoint(time="09:30", price=550),
        PricePoint(time="10:00", price=552),
        PricePoint(time="10:30", price=549),
        PricePoint(time="11:00", price=551),
        PricePoint(time="11:30", price=553),
        PricePoint(time="12:00", price=552),
        PricePoint(time="13:00", price=554),
        PricePoint(time="14:00", price=556),
        PricePoint(time="15:00", price=555),
        PricePoint(time="16:00", price=557),
    ],
    "QQQ": [
        PricePoint(time="09:30", price=480),
        PricePoint(time="10:00", price=482),
        PricePoint(time="10:30", price=479),
        PricePoint(time="11:00", price=481),
        PricePoint(time="11:30", price=485),
        PricePoint(time="12:00", price=487),
        PricePoint(time="13:00", price=489),
        PricePoint(time="14:00", price=492),
        PricePoint(time="15:00", price=491),
        PricePoint(time="16:00", price=493),
    ],
    "TSLA": [
        PricePoint(time="09:30", price=280),
        PricePoint(time="10:00", price=285),
        PricePoint(time="10:30", price=278),
        PricePoint(time="11:00", price=282),
        PricePoint(time="11:30", price=290),
        PricePoint(time="12:00", price=288),
        PricePoint(time="13:00", price=292),
        PricePoint(time="14:00", price=295),
        PricePoint(time="15:00", price=289),
        PricePoint(time="16:00", price=294),
    ],
    "AAPL": [
        PricePoint(time="09:30", price=210),
        PricePoint(time="10:00", price=211),
        PricePoint(time="10:30", price=209),
        PricePoint(time="11:00", price=212),
        PricePoint(time="11:30", price=213),
        PricePoint(time="12:00", price=214),
        PricePoint(time="13:00", price=215),
        PricePoint(time="14:00", price=216),
        PricePoint(time="15:00", price=215),
        PricePoint(time="16:00", price=217),
    ],
}

MOCK_TRADE_IDEAS: List[TradeIdea] = [
    TradeIdea(
        id="1",
        symbol="SPY",
        title="SPY mean-reversion long into VWAP",
        thesis=(
            "Dealer gamma flips mildly positive intraday while liquidity "
            "pocket sits just below spot. Expect fade of morning extension "
            "back into VWAP."
        ),
        direction="LONG",
        confidence=0.72,
        horizon="INTRADAY",
    ),
    TradeIdea(
        id="2",
        symbol="SPY",
        title="0DTE call spread into resistance",
        thesis=(
            "Top of field cone aligns with prior RTH high. Structure: tight "
            "call spread with defined risk, targeting 30–40% move."
        ),
        direction="LONG",
        confidence=0.65,
        horizon="INTRADAY",
    ),
    TradeIdea(
        id="3",
        symbol="TSLA",
        title="TSLA short gamma squeeze potential",
        thesis=(
            "Heavy short-dated call open interest above spot with dealers "
            "short gamma. If spot breaks trigger level, accelerate to next "
            "liquidity shelf."
        ),
        direction="LONG",
        confidence=0.80,
        horizon="SWING",
    ),
    TradeIdea(
        id="4",
        symbol="AAPL",
        title="AAPL covered call candidate",
        thesis=(
            "Vol surface elevated vs realized. Good candidate for covered "
            "calls or call credit spreads at upper field boundary."
        ),
        direction="SHORT",
        confidence=0.60,
        horizon="POSITION",
    ),
    TradeIdea(
        id="5",
        symbol="QQQ",
        title="QQQ downside hedge via put spread",
        thesis=(
            "Macro regime flagging elevated downside tail risk next 5–10 "
            "sessions. Cheap put spreads provide asymmetric protection."
        ),
        direction="SHORT",
        confidence=0.70,
        horizon="SWING",
    ),
]


# -----------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------

@app.get("/api/symbols", response_model=List[SymbolInfo])
def list_symbols() -> List[SymbolInfo]:
    return list(MOCK_SYMBOLS.values())


@app.get("/api/info/{symbol}", response_model=SymbolInfo)
def get_info(symbol: str) -> SymbolInfo:
    return MOCK_SYMBOLS.get(symbol.upper(), list(MOCK_SYMBOLS.values())[0])


@app.get("/api/price-series/{symbol}", response_model=List[PricePoint])
def get_price_series(symbol: str) -> List[PricePoint]:
    return MOCK_PRICE_SERIES.get(symbol.upper(), [])


@app.get("/api/trade-ideas/{symbol}", response_model=List[TradeIdea])
def get_trade_ideas(symbol: str) -> List[TradeIdea]:
    """
    Trade ideas for the selected symbol.

    First attempt: real Gnosis engine via run_full_pipeline_for_symbol.
    Fallback: mock ideas for that symbol if engine import or execution fails.
    """
    symbol = symbol.upper()

    gnosis_ideas = generate_trade_ideas_from_gnosis(symbol)
    uw_ideas = generate_trade_ideas_from_unusual_whales(symbol)

    combined: List[TradeIdea] = []
    seen_ids = set()
    for idea in gnosis_ideas + uw_ideas:
        if idea.id in seen_ids:
            continue
        seen_ids.add(idea.id)
        combined.append(idea)

    combined.sort(key=lambda ti: ti.confidence, reverse=True)
    return combined
