"""
Minimal Market Data Pipeline for your main environment
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class MinimalMarketPipeline:
    """Complete market data pipeline with Yahoo Finance, Fyers, and database storage"""
    
    def __init__(self):
        # Database path in your project
        self.db_path = "financial/data/financial.db"
        self.cache_dir = "financial/data/cache"
        
        # Create directories
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check what's available
        self.sources = self._check_sources()
        print(f"📊 Detected sources: {', '.join(self.sources)}")
        
        # Initialize database
        self._init_database()
    
    def _check_sources(self) -> List[str]:
        """Check which data sources are available"""
        available = []
        
        try:
            import yfinance
            available.append("yfinance")
            print("✅ Yahoo Finance available")
        except ImportError:
            print("⚠️ yfinance not installed")
        
        try:
            from fyers_apiv3 import fyersModel
            available.append("fyers_legacy")
            print("✅ Fyers (Legacy SDK) available")
        except ImportError:
            print("⚠️ fyers-apiv3 not installed")
        
        # Try new Fyers integration if available
        try:
            from auth.fyers_auth import FyersAuth
            auth = FyersAuth()
            if auth.is_authenticated():
                available.append("fyers_new")
                print("✅ Fyers (New Integration) available and authenticated")
            else:
                print("⚠️ Fyers (New Integration) available but not authenticated")
        except ImportError:
            pass  # New integration not available
        
        return available
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT,
            interval TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp, source, interval)
        )
        ''')
        
        # Portfolio positions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_date DATETIME NOT NULL,
            current_price REAL,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            expires_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    # ==================== Upstox ====================

    def fetch_upstox(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Upstox"""
        try:
            from financial.auth.upstox_auth import get_upstox_client
            import upstox_client
        
            client = get_upstox_client()
            if not client:
                return None
        
            # Format symbol for Upstox
            instrument_key = self._format_upstox_symbol(symbol)
            
            # If None, symbol not available on Upstox
            if instrument_key is None:
                print(f"ℹ️ {symbol} not available on Upstox (non-Indian symbol)")
                return None
        
            # Get quote
            market_api = upstox_client.MarketQuoteApi(client)
            response = market_api.get_full_market_quote(
                symbol=instrument_key,
                api_version='2.0'
            )
        
            # Find the correct key in response
            data_key = None
            for key in response.data.keys():
                if symbol.upper() in key or (instrument_key and instrument_key.split('|')[-1] in key):
                    data_key = key
                    break
            
            if not data_key:
                # Try first key as fallback
                data_key = list(response.data.keys())[0] if response.data else None
            
            if not data_key:
                return None
        
            data = response.data[data_key]
            ohlc = data.ohlc
        
            return {
                "symbol": symbol,
                "source": "upstox",
                "interval": "1d",
                "latest_price": data.last_price,
                "data": [{
                    "timestamp": datetime.now().isoformat(),
                    "open": ohlc.open,
                    "high": ohlc.high,
                    "low": ohlc.low,
                    "close": ohlc.close,
                    "volume": data.volume
                }],
                "record_count": 1,
                "currency": "INR"
            }
        except Exception as e:
            print(f"Upstox error for {symbol}: {type(e).__name__}: {str(e)[:100]}")
            return None

    def _format_upstox_symbol(self, symbol: str) -> str:
        """Convert symbol to Upstox format - IMPROVED"""
        symbol = symbol.upper().strip()
        
        # Remove Yahoo prefix
        if symbol.startswith("^"):
            symbol = symbol[1:]
        
        # Remove .NS suffix
        if symbol.endswith(".NS"):
            symbol = symbol[:-3]
        
        # Remove -EQ, -INDEX suffixes
        if symbol.endswith("-EQ"):
            symbol = symbol[:-3]
        elif symbol.endswith("-INDEX"):
            symbol = symbol[:-6]
        
        # Special mappings
        upstox_map = {
            'NIFTY': 'NSE_INDEX|Nifty 50',
            'BANKNIFTY': 'NSE_INDEX|Nifty Bank',
            'SENSEX': 'BSE_INDEX|SENSEX',
            'NSEI': 'NSE_INDEX|Nifty 50',  # Yahoo's NSEI
            'NSEBANK': 'NSE_INDEX|Nifty Bank',  # Yahoo's NSEBANK
            'BSESN': 'BSE_INDEX|SENSEX',  # Yahoo's BSESN
            'RELIANCE': 'NSE_EQ|INE002A01018',
            'TCS': 'NSE_EQ|INE467B01029',
            'INFY': 'NSE_EQ|INE009A01021',
            'HDFCBANK': 'NSE_EQ|INE040A01026',
            'ICICIBANK': 'NSE_EQ|INE090A01021',
            'AAPL': None,  # Skip - not available on Upstox
            'TSLA': None,  # Skip - not available on Upstox
            'GOOGL': None,  # Skip - not available on Upstox
        }
        
        if symbol in upstox_map:
            return upstox_map[symbol]
        
        # Default to NSE equity for Indian symbols, None for others
        if self._is_indian_symbol(symbol):
            return f'NSE_EQ|{symbol}'
        else:
            return None  # Skip Upstox for non-Indian symbols
    
    # ==================== YAHOO FINANCE ====================
    
    def fetch_yfinance(self, symbol: str, period: str = "1mo", 
                    interval: str = "1d") -> Optional[Dict]:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf
            import warnings
            
            # Suppress warnings
            warnings.filterwarnings('ignore', message='possibly delisted')
            
            print(f"📊 Yahoo Finance: Fetching {symbol}...")
            
            # Map symbols
            symbol_map = {
                "NIFTY": "^NSEI",
                "BANKNIFTY": "^NSEBANK", 
                "SENSEX": "^BSESN",
                "NIFTY-INDEX": "^NSEI",
                "BANKNIFTY-INDEX": "^NSEBANK",
                "^GSPC": "^GSPC",
                "GSPC": "^GSPC",
            }
            
            yahoo_symbol = symbol_map.get(symbol, symbol)
            
            # If Indian symbol without suffix and not an index, add .NS
            if (self._is_indian_symbol(symbol) and 
                not any(x in yahoo_symbol for x in ['.', '^', '-']) and
                "NIFTY" not in symbol and "SENSEX" not in symbol):
                yahoo_symbol = yahoo_symbol + ".NS"
            
            ticker = yf.Ticker(yahoo_symbol)
            
            # FIX: Remove progress parameter for newer yfinance versions
            try:
                # Try without progress parameter
                df = ticker.history(period=period, interval=interval)
            except TypeError:
                # Fallback for older versions
                df = ticker.history(period=period, interval=interval, progress=False)
            
            if df.empty:
                print(f"   No data for {symbol} (tried as {yahoo_symbol})")
                return None
            
            # Convert to records
            records = []
            for idx, row in df.iterrows():
                records.append({
                    "timestamp": idx.isoformat(),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0))
                })
            
            result = {
                "symbol": symbol,
                "source": "yfinance",
                "interval": interval,
                "data": records,
                "latest_price": float(df["Close"].iloc[-1]),
                "record_count": len(records),
                "currency": "INR" if ".NS" in yahoo_symbol or yahoo_symbol in ["^NSEI", "^NSEBANK", "^BSESN"] else "USD"
            }
            
            # Store in database
            self._store_market_data(result)
            
            print(f"✅ {symbol}: {len(records)} records, Latest: {result['latest_price']:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ Yahoo Finance error for {symbol}: {e}")
            return None
    
    # ==================== FYERS ====================
    
    def _get_fyers_token(self) -> Optional[str]:
        """Get Fyers token from your existing setup"""
        # Try multiple possible locations
        token_files = [
            "financial/auth/fyers_token.json",
            "financial/auth/fyers_token.txt",
            "fyers_token.json",
            "fyers_token.txt",
            os.path.join(os.path.expanduser("~"), "fyers_token.json"),
            os.path.join(os.getcwd(), "fyers_token.txt")
        ]
        
        for token_file in token_files:
            if os.path.exists(token_file):
                try:
                    with open(token_file, 'r') as f:
                        content = f.read().strip()
                        if content.startswith('{'):
                            data = json.loads(content)
                            token = data.get("access_token") or data.get("token")
                        else:
                            token = content
                        
                        if token:
                            print(f"✓ Using Fyers token from: {token_file}")
                            return token
                except Exception as e:
                    print(f"  Warning reading {token_file}: {e}")
                    continue
        
        # Check environment variable
        token = os.getenv("FYERS_TOKEN")
        if token:
            print("✓ Using Fyers token from environment")
            return token
        
        print("⚠️ No Fyers token found")
        return None
    
    def fetch_fyers(self, symbol: str, interval: str = "D") -> Optional[Dict]:
        """Fetch data from Fyers API"""
        try:
            from fyers_apiv3 import fyersModel
            
            token = self._get_fyers_token()
            if not token:
                return None
            
            app_id = os.getenv("FYERS_APP_ID")
            if not app_id:
                print("⚠️ FYERS_APP_ID not set in environment")
                return None
            
            # Create Fyers client
            fyers = fyersModel.FyersModel(
                client_id=app_id,
                token=token,
                log_path="",
                is_async=False
            )
            
            print(f"📊 Fyers: Fetching {symbol}...")
            
            # Format symbol for Fyers
            fyers_symbol = self._format_fyers_symbol(symbol)
            
            # Date range (last 30 days)
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            response = fyers.history({
                "symbol": fyers_symbol,
                "resolution": interval,
                "date_format": "1",
                "range_from": from_date,
                "range_to": to_date,
                "cont_flag": "1"
            })
            
            if response.get("s") != "ok":
                print(f"   Fyers API error: {response.get('message', 'Unknown error')}")
                return None
            
            candles = response.get("candles", [])
            if not candles:
                print(f"   No candle data for {symbol}")
                return None
            
            # Convert to records
            records = []
            for candle in candles:
                # Convert timestamp (milliseconds to seconds)
                ts = candle[0]
                if ts > 1e10:  # If in milliseconds
                    ts = ts / 1000
                
                records.append({
                    "timestamp": datetime.fromtimestamp(ts).isoformat(),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]) if len(candle) > 5 else 0
                })
            
            result = {
                "symbol": symbol,
                "source": "fyers",
                "interval": interval,
                "data": records,
                "latest_price": float(candles[-1][4]),
                "record_count": len(records),
                "currency": "INR"
            }
            
            # Store in database
            self._store_market_data(result)
            
            print(f"✅ {symbol}: {len(records)} records, Latest: ₹{result['latest_price']:.2f}")
            return result
            
        except ImportError:
            print("⚠️ Fyers SDK not installed")
            return None
        except Exception as e:
            print(f"❌ Fyers error for {symbol}: {e}")
            return None
    
    def _format_fyers_symbol(self, symbol: str) -> str:
        """Format symbol for Fyers API"""
        if ":" not in symbol:
            if symbol.endswith("-EQ"):
                return f"NSE:{symbol}"
            elif "NIFTY" in symbol or "BANKNIFTY" in symbol:
                return f"NSE:{symbol}-INDEX"
            elif symbol.startswith("^"):
                # Remove ^ and add -INDEX if needed
                clean = symbol.replace("^", "")
                if "NIFTY" in clean or "BANKNIFTY" in clean:
                    return f"NSE:{clean}-INDEX"
                else:
                    return f"NSE:{clean}-EQ"
            else:
                # Default to NSE equity
                return f"NSE:{symbol}-EQ"
        return symbol
    
    # ==================== UNIFIED FETCH METHODS ====================
    
    def choose_data_source(self, symbol: str, interval: str = "1d") -> List[str]:
        """
        Smart data source selection with proper fallback sequence
        Brokers (Upstox/Fyers) can provide both live and historical data
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')
        
        Returns:
            List of sources in priority order
        """
        symbol = symbol.upper().strip()
        
        # Check if it's an Indian symbol
        is_indian = self._is_indian_symbol(symbol)
        
        # Determine if we need real-time capabilities
        is_realtime_interval = interval in ["1m", "5m", "15m", "30m", "1h"]
        is_historical_interval = interval in ["1d", "1wk", "1mo", "3mo", "6mo", "1y"]
        
        print(f"🔍 Source selection for {symbol}: interval={interval}, indian={is_indian}")
        print(f"   Real-time interval: {is_realtime_interval}")
        print(f"   Historical interval: {is_historical_interval}")
        
        # ========== INDIAN SYMBOLS ==========
        if is_indian:
            if is_realtime_interval:
                # For real-time Indian data: Brokers first (they have best real-time)
                return ["upstox", "fyers", "yfinance"]
            elif is_historical_interval:
                # For historical Indian data: Try brokers first as they might have cleaner data
                return ["upstox", "fyers", "yfinance"]
            else:
                # Default for other intervals
                return ["upstox", "fyers", "yfinance"]
        
        # ========== GLOBAL SYMBOLS ==========
        else:
            if is_realtime_interval:
                # For real-time global: Yahoo Finance has good real-time for global
                return ["yfinance", "upstox", "fyers"]
            elif is_historical_interval:
                # For historical global: Yahoo Finance is best
                return ["yfinance", "upstox", "fyers"]
            else:
                # Default for global
                return ["yfinance", "upstox", "fyers"]

    def _is_indian_symbol(self, symbol: str) -> bool:
        """Check if symbol is Indian with improved logic"""
        symbol = symbol.upper().strip()
        
        # Remove common suffixes for checking
        base_symbol = symbol
        suffixes_to_remove = [".NS", ".BO", ".NSE", ".BSE", "-EQ", "-INDEX", "-BE", "-NE"]
        
        for suffix in suffixes_to_remove:
            if symbol.endswith(suffix):
                base_symbol = symbol[:-len(suffix)]
                break
        
        # Known Indian stocks and indices
        known_indian_stocks = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "HDFC", "ICICIBANK", "KOTAKBANK",
            "AXISBANK", "SBIN", "ITC", "LT", "BAJFINANCE", "BHARTIARTL", "ASIANPAINT",
            "HINDUNILVR", "MARUTI", "TITAN", "SUNPHARMA", "TATAMOTORS", "WIPRO",
            "ONGC", "POWERGRID", "NTPC", "ULTRACEMCO", "M&M", "BAJAJ-AUTO", "NESTLE",
            "TECHM", "HCLTECH", "DRREDDY", "BRITANNIA", "DIVISLAB", "EICHERMOT",
            "GRASIM", "JSWSTEEL", "TATASTEEL", "ADANIPORTS", "IOC", "BPCL", "HINDALCO",
            "COALINDIA", "VEDANTA", "SHREECEM", "UPL", "ZEEL"
        ]
        
        known_indian_indices = [
            "NIFTY", "BANKNIFTY", "SENSEX", "NIFTY50", "NIFTYBANK", "NIFTYIT",
            "NIFTYMIDCAP", "NIFTYSMALLCAP", "SENSEX30", "BSE500"
        ]
        
        # KNOWN GLOBAL SYMBOLS (should NOT be Indian)
        known_global_symbols = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", 
            "BRK.B", "BRK.A", "JPM", "JNJ", "V", "UNH", "XOM", "WMT", "PG",
            "MA", "HD", "CVX", "BAC", "PFE", "KO", "ABBV", "AVGO", "COST",
            "DIS", "CSCO", "PEP", "ADBE", "TMO", "ACN", "DHR", "NFLX", "MCD",
            "CRM", "ABT", "LIN", "NKE", "PM", "TXN", "CMCSA", "AMD", "HON",
            "QCOM", "INTU", "INTC", "IBM", "GS", "CAT", "UPS", "RTX", "BA",
            "UNP", "PLD", "T", "DE", "SPGI", "LOW", "SBUX", "ISRG", "LMT",
            "MDT", "BMY", "GE", "AMT", "BLK", "ELV", "C", "VZ", "PYPL", "GILD",
            "MO", "AXP", "SCHW", "NOW", "ADI", "BKNG", "MMM", "CI", "TJX",
            "CVS", "CB", "MDLZ", "DUK", "SO", "ZTS", "REGN", "TGT", "BDX",
            "PNC", "BSX", "EOG", "NOC", "USB", "CL", "COP", "AMGN", "SHW",
            "SLB", "FISV", "ICE", "NSC", "PGR", "EQIX", "ORCL", "HCA", "APD",
            "MCK", "FDX", "EMR", "PSX", "GD", "AON", "MET", "NEM", "KMB",
            "GM", "MRNA", "DG", "SRE", "MAR", "APO", "WM", "KLAC", "CTAS",
            "ADP", "GIS", "CME", "SNPS", "ECL", "AEP", "AIG", "CCI", "LHX",
            "PH", "AJG", "TDG", "MS", "SPG", "ITW", "NUE", "TRV", "OXY",
            "DOW", "STZ", "VRTX", "EW", "AZO", "WELL", "CDNS", "CSX", "ADSK",
            "ROST", "MTB", "MCO", "IDXX", "ORLY", "PCAR", "MNST", "AFL",
            "DLR", "ROP", "PPG", "PAYX", "ALL", "YUM", "RSG", "F", "TFC",
            "DHI", "SYY", "WFC", "GLW", "EL", "WMB", "AME", "TT", "IQV",
            "HUM", "PSA", "EA", "ANET", "RMD", "VLO", "CTSH", "PRU", "HLT",
            "PEG", "HPQ", "FTNT", "MTD", "ROK", "FAST", "MLM", "SWKS", "DD",
            "KMI", "OTIS", "KHC", "WBD", "LYB", "AVB", "BIIB", "OKE", "EXC",
            "IR", "SBAC", "KR", "DAL", "EFX", "NDAQ", "HAL", "ES", "ALB",
            "EBAY", "VMC", "CPRT", "XEL", "FTV", "MPWR", "TSCO", "GRMN",
            "DFS", "AWK", "KEYS", "GWW", "ED", "WST", "CBRE", "APH",
            "TTWO", "STT", "RJF", "WEC", "EQR", "ZBH", "DTE", "VRSK", "ULTA",
            "ETN", "ARE", "CTVA", "WTW", "GNRC", "FANG", "LVS", "DOV", "PFG",
            "D", "DLTR", "WAB", "LYV", "IRM", "EXR", "WY", "PAYC", "BR",
            "LEN", "HBAN", "MSCI", "INVH", "CFG", "HPE", "NTAP", "VTR",
            "HIG", "LUV", "MTCH", "STX", "CZR", "LRCX", "CDW", "NVR", "ROL",
            "CMS", "OMC", "HSY", "DGX", "KEY", "CNC", "BXP", "MKC", "NI",
            "MAA", "FE", "RF", "KMX", "HWM", "AVY", "PTC", "SNA", "WRB",
            "STE", "ESS", "K", "CHTR", "COO", "AKAM", "SJM", "SYF",
            "J", "TDY", "EXPD", "WDC", "CINF", "LH", "BWA", "REG", "TPR",
            "FBHS", "SWK", "CLX", "EPAM", "UAL", "NDSN", "ARES", "MOS",
            "POOL", "TROW", "MAS", "AEE", "FLT", "IFF", "CE", "TXT", "PNR",
            "XYL", "DISH", "HST", "IP", "RCL", "TER", "BBY", "UDR", "JKHY",
            "CPB", "HOLX", "JNPR", "NWSA", "GL", "NTRS", "DRI", "QRVO",
            "WHR", "EIX", "NRG", "LKQ", "IVZ", "TFX", "HRL", "ATO", "BEN",
            "L", "PNW", "CPT", "AIZ", "AES", "BRO", "CMA", "TAP", "ZBRA",
            "CAG", "RE", "AAP", "FITB", "WAT", "IPG", "ETR", "HAS", "FFIV",
            "VFC", "PKI", "NWL", "PBCT", "WU", "LDOS", "EMN", "RL", "JEF",
            "ALK", "UHS", "BALL", "CTRA", "PVH", "MRO", "HBI", "FOX", "NCLH",
            "LW", "LNC", "NLOK", "BF.B", "VNO", "CCL", "DISCA", "DXC", "KSS",
            "NLSN", "SEE", "GPS", "FRT", "MGM", "LEG", "SLG", "DVA", "OGN",
            "FOXA", "HRB", "BTC", "ETH"
        ]
        
        # Check if it's a known Indian stock
        if base_symbol in known_indian_stocks:
            return True
        
        # Check if it's a known Indian index
        if base_symbol in known_indian_indices:
            return True
        
        # Check if it's a known GLOBAL stock (NOT Indian)
        if base_symbol in known_global_symbols:
            return False
        
        # Check by pattern (NSE format)
        if symbol.startswith("NSE:") or symbol.startswith("BSE:"):
            return True
        
        # Check by suffix
        if any(symbol.endswith(suffix) for suffix in suffixes_to_remove):
            return True
        
        # Check if it's a US index (starts with ^)
        if symbol.startswith("^"):
            return False
        
        # Check if it's a cryptocurrency
        if "-USD" in symbol or "-INR" in symbol or symbol in ["BTC", "ETH", "XRP", "ADA"]:
            return False
        
        # Check if it looks like an NSE/BSE symbol (typically short, alphanumeric)
        if 2 <= len(base_symbol) <= 20 and base_symbol.isalnum():
            # Could be Indian, check with additional patterns
            if base_symbol.isalpha() and base_symbol.isupper():
                # Default to not Indian unless we have evidence otherwise
                # Most Indian symbols are 2-8 chars, US can be longer
                if len(base_symbol) <= 8:
                    return True  # Short uppercase - likely Indian
                else:
                    return False  # Longer symbols more likely global
        
        # Default to not Indian
        return False
    
    def fetch_market_data(self, symbol: str, source: str = "auto", 
                          interval: str = "1d") -> Optional[Dict]:
        """
        Unified method to fetch market data from any source
        """
        symbol = symbol.replace(".NS", "").replace("^", "")
        symbol = symbol.strip().upper()

        # >>> FIX START: explicit source handling (ONLY CHANGE)
        if source == "upstox":
            return self.fetch_upstox(symbol)

        if source == "fyers":
            return self._fetch_unified_fyers(symbol, interval)

        if source == "yfinance":
            period = "5d" if interval in ["1min", "5min"] else "1mo"
            return self.fetch_yfinance(symbol, period=period, interval=interval)
        # <<< FIX END

        # Auto-select source
        if source == "auto":
            # Use the new smart source selection
            order = self.choose_data_source(symbol, interval)

            for src in order:
                if src == "upstox":
                    data = self.fetch_upstox(symbol)
                elif src == "fyers":
                    data = self._fetch_unified_fyers(symbol, interval)
                elif src == "yfinance":
                    period = "5d" if interval in ["1min", "5min"] else "1mo"
                    data = self.fetch_yfinance(symbol, period=period, interval=interval)
                else:
                    data = None

                if data:
                    return data

            print(f"⚠️ All data sources failed for {symbol}")
            return None


    def _fetch_unified_fyers(self, symbol: str, interval: str = "D") -> Optional[Dict]:
        """Try multiple Fyers integrations"""
        # Map interval for Fyers
        interval_map = {
            "1d": "D", "1h": "60", "30min": "30", "15min": "15",
            "5min": "5", "1min": "1"
        }
        fyers_interval = interval_map.get(interval, "D")
        
        
        # Fallback to legacy SDK
        if "fyers_legacy" in self.sources:
            return self.fetch_fyers(symbol, fyers_interval)
    
        print("⚠️ Fyers integration not available")
        return None
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get latest live price"""
        data = self.fetch_market_data(symbol, source="auto")
        return data.get("latest_price") if data else None
    
    # ==================== DATABASE METHODS ====================
    
    def _store_market_data(self, data: Dict):
        """Store market data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for record in data.get("data", []):
                cursor.execute('''
                INSERT OR IGNORE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, source, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data["symbol"],
                    record["timestamp"],
                    record.get("open"),
                    record.get("high"),
                    record.get("low"),
                    record.get("close"),
                    record.get("volume", 0),
                    data["source"],
                    data.get("interval", "1d")
                ))
            
            conn.commit()
            print(f"💾 Stored {len(data.get('data', []))} records for {data['symbol']}")
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            conn.close()
    
    def get_historical_data(self, symbol: str, days: int = 30, 
                           source: str = "auto") -> List[Dict]:
        """Get historical data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
        SELECT timestamp, open, high, low, close, volume, source, interval
        FROM market_data 
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        ''', (symbol, cutoff_date))
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "source": row[6],
                "interval": row[7]
            })
        
        conn.close()
        return records
    
    # ==================== PORTFOLIO METHODS ====================
    
    def add_portfolio_position(self, symbol: str, quantity: float, 
                               entry_price: float, notes: str = ""):
        """Add a position to portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO portfolio (symbol, quantity, entry_price, entry_date, notes)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                quantity,
                entry_price,
                datetime.now().isoformat(),
                notes
            ))
            
            conn.commit()
            print(f"✅ Added {quantity} shares of {symbol} at {entry_price:.2f}")
        except Exception as e:
            print(f"Portfolio error: {e}")
        finally:
            conn.close()
    
    def update_portfolio_position(self, position_id: int, **kwargs):
        """Update portfolio position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            update_fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in ["quantity", "entry_price", "current_price", "notes"]:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            if update_fields:
                values.append(position_id)
                cursor.execute(f'''
                UPDATE portfolio 
                SET {', '.join(update_fields)}
                WHERE id = ?
                ''', values)
                
                conn.commit()
                print(f"✅ Updated position {position_id}")
        except Exception as e:
            print(f"Update portfolio error: {e}")
        finally:
            conn.close()
    
    def get_portfolio_positions(self) -> List[Dict]:
        """Get all portfolio positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM portfolio WHERE quantity > 0')
        positions = cursor.fetchall()
        conn.close()
        
        result = []
        for pos in positions:
            result.append({
                "id": pos[0],
                "symbol": pos[1],
                "quantity": pos[2],
                "entry_price": pos[3],
                "entry_date": pos[4],
                "current_price": pos[5],
                "notes": pos[6],
                "created_at": pos[7]
            })
        
        return result
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with current values"""
        positions = self.get_portfolio_positions()
        
        if not positions:
            return {
                "total_value": 0,
                "total_invested": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "positions": []
            }
        
        summary = {
            "total_invested": 0,
            "total_current": 0,
            "total_pnl": 0,
            "positions": []
        }
        
        for pos in positions:
            symbol = pos["symbol"]
            quantity = pos["quantity"]
            entry_price = pos["entry_price"]
            
            # Get current price
            current_price = self.get_live_price(symbol)
            if not current_price:
                current_price = pos.get("current_price", entry_price)
            
            invested = quantity * entry_price
            current_value = quantity * current_price
            pnl = current_value - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0
            
            position_summary = {
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "invested": invested,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "notes": pos.get("notes", "")
            }
            
            summary["positions"].append(position_summary)
            summary["total_invested"] += invested
            summary["total_current"] += current_value
        
        summary["total_pnl"] = summary["total_current"] - summary["total_invested"]
        summary["total_pnl_pct"] = (summary["total_pnl"] / summary["total_invested"] * 100) if summary["total_invested"] > 0 else 0
        
        return summary
    
    # ==================== CACHE METHODS ====================
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT value, expires_at FROM cache 
        WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
        ''', (key, datetime.now().isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                return json.loads(row[0])
            except:
                return row[0]
        
        return None
    
    def _set_cache(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        expires_at = None
        if ttl_seconds > 0:
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        
        try:
            json_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            
            cursor.execute('''
            INSERT OR REPLACE INTO cache (key, value, expires_at)
            VALUES (?, ?, ?)
            ''', (key, json_value, expires_at))
            
            conn.commit()
        except Exception as e:
            print(f"Cache error: {e}")
        finally:
            conn.close()
    
    # ==================== HELPER METHODS ====================
    
    def get_multi_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get prices for multiple symbols"""
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_live_price(symbol)
        return result
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get information about a symbol"""
        data = self.fetch_market_data(symbol)
        if not data:
            return None
        
        return {
            "symbol": symbol,
            "current_price": data.get("latest_price"),
            "currency": data.get("currency", "USD"),
            "source": data.get("source"),
            "data_points": len(data.get("data", [])),
            "last_updated": datetime.now().isoformat()
        }
    
    def search_symbols(self, query: str) -> List[Dict]:
        """Search for symbols (simplified - in production would use API)"""
        # Common symbols for quick search
        common_symbols = {
            "AAPL": {"name": "Apple Inc.", "type": "stock", "exchange": "NASDAQ"},
            "MSFT": {"name": "Microsoft Corporation", "type": "stock", "exchange": "NASDAQ"},
            "GOOGL": {"name": "Alphabet Inc.", "type": "stock", "exchange": "NASDAQ"},
            "NIFTY": {"name": "Nifty 50", "type": "index", "exchange": "NSE"},
            "RELIANCE": {"name": "Reliance Industries", "type": "stock", "exchange": "NSE"},
            "TCS": {"name": "Tata Consultancy Services", "type": "stock", "exchange": "NSE"},
            "BTC-USD": {"name": "Bitcoin", "type": "crypto", "exchange": "Crypto"},
            "^GSPC": {"name": "S&P 500", "type": "index", "exchange": "NYSE"}
        }
        
        query = query.upper()
        results = []
        
        for symbol, info in common_symbols.items():
            if query in symbol or query in info["name"].upper():
                results.append({
                    "symbol": symbol,
                    **info,
                    "suggested_symbol": f"{symbol}.NS" if info["exchange"] == "NSE" and "-EQ" not in symbol else symbol
                })
        
        return results[:10]  # Limit results


# ==================== TEST FUNCTION ====================

def test_pipeline():
    """Test the pipeline"""
    print("🧪 Testing Complete Market Pipeline")
    print("="*50)
    
    # Initialize
    pipeline = MinimalMarketPipeline()
    
    # Test 1: Yahoo Finance
    print("\n1. Testing Yahoo Finance...")
    us_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in us_symbols:
        data = pipeline.fetch_market_data(symbol, source="yfinance")
        if data:
            print(f"   ✅ {symbol}: {data['latest_price']:.2f} ({data['source']})")
        else:
            print(f"   ❌ {symbol}: Failed")
    
    # Test 2: Fyers (if available)
    print("\n2. Testing Fyers...")
    indian_symbols = ["NIFTY50-INDEX", "RELIANCE-EQ"]
    
    for symbol in indian_symbols:
        data = pipeline.fetch_market_data(symbol, source="fyers")
        if data:
            print(f"   ✅ {symbol}: {data['latest_price']:.2f} ({data['source']})")
        else:
            print(f"   ❌ {symbol}: Failed or not authenticated")
    
    # Test 3: Auto detection
    print("\n3. Testing Auto Source Detection...")
    test_symbols = ["AAPL", "NIFTY", "TCS.NS", "BTC-USD"]
    
    for symbol in test_symbols:
        data = pipeline.fetch_market_data(symbol, source="auto")
        if data:
            print(f"   ✅ {symbol}: {data['latest_price']:.2f} via {data['source']}")
        else:
            print(f"   ❌ {symbol}: Failed")
    
    # Test 4: Portfolio
    print("\n4. Testing Portfolio...")
    # Add test positions
    pipeline.add_portfolio_position("AAPL", 10, 150.25, "Test position")
    pipeline.add_portfolio_position("MSFT", 5, 300.50, "Another test")
    
    portfolio = pipeline.get_portfolio_summary()
    print(f"   Total positions: {len(portfolio['positions'])}")
    print(f"   Portfolio value: {portfolio['total_current']:.2f}")
    print(f"   Total P&L: {portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.1f}%)")
    
    # Test 5: Historical data
    print("\n5. Testing Historical Data...")
    historical = pipeline.get_historical_data("AAPL", days=7)
    print(f"   AAPL historical records: {len(historical)}")
    
    # Test 6: Multi prices
    print("\n6. Testing Multi Prices...")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    prices = pipeline.get_multi_prices(symbols)
    for sym, price in prices.items():
        if price:
            print(f"   {sym}: {price:.2f}")
        else:
            print(f"   {sym}: N/A")
    
    # Test 7: Symbol search
    print("\n7. Testing Symbol Search...")
    results = pipeline.search_symbols("apple")
    for result in results:
        print(f"   Found: {result['symbol']} - {result['name']}")
    
    print("\n" + "="*50)
    print("✅ All tests completed!")
    
    return pipeline


if __name__ == "__main__":
    test_pipeline()