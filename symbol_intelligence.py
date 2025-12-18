# symbol_intelligence.py

# STEP 2 — data contract
from typing import TypedDict, Optional
import re

class ResolvedInstrument(TypedDict):
    asset_class: str
    symbol: str
    confidence: float


# STEP 3 — extract raw symbol (THIS ANSWERS YOUR QUESTION)
def extract_raw_symbol(text: str) -> Optional[str]:
    t = text.lower()

    noise = [
        "price", "trading", "right now", "now",
        "current", "where is", "last", "ltp"
    ]
    for n in noise:
        t = t.replace(n, " ")

    tokens = re.findall(r"[a-zA-Z/]+", t)

    if not tokens:
        return None

    return tokens[0]


# STEP 4 — classify asset class
def classify_asset(raw: str) -> Optional[str]:
    s = raw.lower()

    if "/" in s or (s.isalpha() and len(s) <= 5):
        return "crypto"

    if s in {"nifty", "banknifty", "sensex"}:
        return "index"

    if s.isalpha():
        return "indian_equity"

    return None


# STEP 5 — normalize symbol
def normalize_symbol(asset_class: str, raw: str) -> Optional[str]:
    s = raw.upper()

    if asset_class == "crypto":
        return s if "/" in s else f"{s}/USD"

    if asset_class == "indian_equity":
        return f"NSE:{s}-EQ"

    if asset_class == "index":
        INDEX_MAP = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:BANKNIFTY-INDEX",
            "SENSEX": "BSE:SENSEX-INDEX",
        }
        return INDEX_MAP.get(s)

    return None


# STEP 6 — public resolver (single entry point)
def resolve_instrument(text: str) -> Optional[ResolvedInstrument]:
    raw = extract_raw_symbol(text)
    if not raw:
        return None

    asset_class = classify_asset(raw)
    if not asset_class:
        return None

    symbol = normalize_symbol(asset_class, raw)
    if not symbol:
        return None

    return {
        "asset_class": asset_class,
        "symbol": symbol,
        "confidence": 0.8
    }
