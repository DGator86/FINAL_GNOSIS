import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from loguru import logger

# Try to import the UnusualWhalesAdapter
try:
    from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
except Exception:
    UnusualWhalesAdapter = None  # makes API still run if adapter missing

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
UW_TOKEN = os.getenv("UNUSUAL_WHALES_API_KEY") or os.getenv("UNUSUAL_WHALES_TOKEN")

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

def _coerce_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _parse_direction(flow: dict) -> Optional[str]:
    candidates = [
        flow.get("direction"), flow.get("sentiment"),
        flow.get("side"), flow.get("order_side"),
        flow.get("option_type"), flow.get("type"),
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


# -----------------------------------------------------
# UNUSUAL WHALES PIPELINE
# -----------------------------------------------------

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

    ideas.sort(key=lambda x: x.confidence, reverse=True)
    return ideas or [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]


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
    return generate_trade_ideas_from_unusual_whales(symbol.upper())
