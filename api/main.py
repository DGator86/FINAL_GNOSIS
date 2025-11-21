from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict


app = FastAPI(title="Gnosis Backend API")

# Allow your Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later we can lock this down to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Data Models ----------

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
    direction: str    # "LONG" or "SHORT"
    confidence: float # 0–1
    horizon: str      # "INTRADAY" | "SWING" | "POSITION"


# ---------- Temporary Mock Data (same as frontend) ----------

MOCK_SYMBOLS: Dict[str, SymbolInfo] = {
    "SPY": SymbolInfo(
        symbol="SPY",
        name="SPDR S&P 500 ETF",
        sector="Index / ETF",
        marketCap="$500B+",
        beta=1.0,
        description=(
            "Proxy for broad U.S. large-cap market. "
            "Often used as the primary field reference for Gnosis."
        ),
    ),
    "QQQ": SymbolInfo(
        symbol="QQQ",
        name="Invesco QQQ Trust",
        sector="Index / ETF – Tech heavy",
        marketCap="$250B+",
        beta=1.2,
        description=(
            "Nasdaq-100 tracker, higher beta vs SPY. "
            "Strong tech concentration, good for momentum and rotations."
        ),
    ),
    "TSLA": SymbolInfo(
        symbol="TSLA",
        name="Tesla, Inc.",
        sector="Consumer Discretionary / EV",
        marketCap="$500B+",
        beta=1.8,
        description=(
            "High-beta single name. Excellent vehicle for dealer-hedging "
            "flows and 0DTE structures."
        ),
    ),
    "AAPL": SymbolInfo(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Information Technology",
        marketCap="$3T+",
        beta=1.1,
        description=(
            "Mega-cap, heavy index weight. Critical for SPY/QQQ structure "
            "and institutional positioning."
        ),
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


# ---------- API Endpoints ----------

@app.get("/api/symbols", response_model=List[SymbolInfo])
def list_symbols() -> List[SymbolInfo]:
    """Watchlist symbols."""
    return list(MOCK_SYMBOLS.values())


@app.get("/api/info/{symbol}", response_model=SymbolInfo)
def get_info(symbol: str) -> SymbolInfo:
    symbol = symbol.upper()
    return MOCK_SYMBOLS.get(symbol, list(MOCK_SYMBOLS.values())[0])


@app.get("/api/price-series/{symbol}", response_model=List[PricePoint])
def get_price_series(symbol: str) -> List[PricePoint]:
    symbol = symbol.upper()
    return MOCK_PRICE_SERIES.get(symbol, [])


@app.get("/api/trade-ideas/{symbol}", response_model=List[TradeIdea])
def get_trade_ideas(symbol: str) -> List[TradeIdea]:
    symbol = symbol.upper()
    return [idea for idea in MOCK_TRADE_IDEAS if idea.symbol == symbol]
