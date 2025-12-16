import re

# --- Explicit index mapping ---
INDEX_MAP = {
    "nifty": "NSE:NIFTY50-INDEX",
    "nifty 50": "NSE:NIFTY50-INDEX",
    "banknifty": "NSE:NIFTYBANK-INDEX",
    "bank nifty": "NSE:NIFTYBANK-INDEX",
    "finnifty": "NSE:FINNIFTY-INDEX",
    "fin nifty": "NSE:FINNIFTY-INDEX",
    "sensex": "BSE:SENSEX-INDEX",
}

def resolve_symbol(user_input: str) -> str | None:
    """
    Resolve user input into a FYERS-compatible trading symbol.
    Returns None if no reasonable symbol can be resolved.
    """

    if not user_input or not user_input.strip():
        return None

    text = user_input.lower().strip()

    # 1️⃣ Check explicit indices
    for key, symbol in INDEX_MAP.items():
        if key in text:
            return symbol

    # 2️⃣ Clean input: remove punctuation
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = cleaned.split()

    # 3️⃣ Heuristic: longest alphabetic token is likely the scrip
    candidates = [t.upper() for t in tokens if len(t) >= 3]

    if not candidates:
        return None

    # Pick the longest token (more stable than first)
    equity = max(candidates, key=len)

    return f"NSE:{equity}-EQ"
