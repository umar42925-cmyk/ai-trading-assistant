from datetime import datetime


def get_fyers_model():
    try:
        from fyers_apiv3 import fyersModel
        return fyersModel
    except ImportError as e:
        raise ImportError("fyers-apiv3 not installed") from e

class FyersMarketData:
    def __init__(self, app_id: str, access_token: str):
        if not app_id:
            raise ValueError("APP_ID is required")
        if not access_token:
            raise ValueError("FYERS access token is required")

        fyersModel = get_fyers_model()   # 👈 THIS WAS MISSING

        self.fyers = fyersModel.FyersModel(
         client_id=app_id,
         token=access_token,
         log_path=""
        )


    def fetch_live_quote(self, symbol: str) -> dict:
        try:
            response = self.fyers.quotes({"symbols": symbol})

            if not response or response.get("s") != "ok":
                return {"error": "FETCH_FAILED"}

            data = response.get("d", [])
            if not data:
                return {"error": "SYMBOL_INVALID"}

            quote = data[0].get("v", {})
            if not quote or quote.get("lp") is None:
                return {"error": "FETCH_FAILED"}

            return {
                "symbol": symbol,
                "price": float(quote["lp"]),
                "change": float(quote.get("ch", 0.0)),
                "change_pct": float(quote.get("chp", 0.0)),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

        except Exception:
            return {"error": "FETCH_FAILED"}

