"""
Minimal Market Data Pipeline for your main environment
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class MinimalMarketPipeline:
    """Minimal pipeline for your existing environment"""
    
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
            available.append("fyers")
            print("✅ Fyers available")
        except ImportError:
            print("⚠️ fyers-apiv3 not installed")
        
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
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    # ==================== YAHOO FINANCE ====================
    
    def fetch_yfinance(self, symbol: str, period: str = "1mo", 
                       interval: str = "1d") -> Optional[Dict]:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf
            
            print(f"📊 Yahoo Finance: Fetching {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"   No data for {symbol}")
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
                "record_count": len(records)
            }
            
            # Store in database
            self._store_market_data(result)
            
            print(f"✅ {symbol}: {len(records)} records, Latest: ${result['latest_price']:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ Yahoo Finance error: {e}")
            return None
    
    # ==================== FYERS ====================
    
    def _get_fyers_token(self) -> Optional[str]:
        """Get Fyers token from your existing setup"""
        # Try multiple possible locations
        token_files = [
            "financial/auth/fyers_token.json",
            "fyers_token.json",
            os.path.join(os.path.expanduser("~"), "fyers_token.json"),
            os.path.join(os.getcwd(), "fyers_token.json")
        ]
        
        for token_file in token_files:
            if os.path.exists(token_file):
                try:
                    with open(token_file, 'r') as f:
                        data = json.load(f)
                    
                    token = data.get("access_token")
                    if token:
                        print(f"✓ Using Fyers token from: {token_file}")
                        return token
                except Exception as e:
                    continue
        
        print("⚠️ No Fyers token found")
        return None
    
    def fetch_fyers(self, symbol: str, interval: str = "D") -> Optional[Dict]:
        """Fetch data from Fyers"""
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
                log_path=""
            )
            
            print(f"📊 Fyers: Fetching {symbol}...")
            
            # Format symbol
            fyers_symbol = self._format_fyers_symbol(symbol)
            
            # Date range
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
                print(f"   Fyers API error: {response.get('message', 'Unknown')}")
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
                    "volume": float(candle[5])
                })
            
            result = {
                "symbol": symbol,
                "source": "fyers",
                "interval": interval,
                "data": records,
                "latest_price": float(candles[-1][4]),
                "record_count": len(records)
            }
            
            # Store in database
            self._store_market_data(result)
            
            print(f"✅ {symbol}: {len(records)} records, Latest: ₹{result['latest_price']:.2f}")
            return result
            
        except Exception as e:
            print(f"❌ Fyers error: {e}")
            return None
    
    def _format_fyers_symbol(self, symbol: str) -> str:
        """Format symbol for Fyers API"""
        if ":" not in symbol:
            if symbol.endswith("-EQ"):
                return f"NSE:{symbol}"
            elif "NIFTY" in symbol or "BANKNIFTY" in symbol:
                return f"NSE:{symbol}-INDEX"
            else:
                # Default to NSE equity
                return f"NSE:{symbol}-EQ"
        return symbol
    
    # ==================== UNIFIED METHODS ====================
    
    def fetch_market_data(self, symbol: str, source: str = "auto") -> Optional[Dict]:
        """
        Fetch market data with automatic source selection
        """
        # Auto-select source
        if source == "auto":
            # Indian symbols → try Fyers first
            is_indian = any(indicator in symbol.upper() 
                          for indicator in ["NIFTY", "BANKNIFTY", "-EQ"])
            
            if is_indian and "fyers" in self.sources:
                data = self.fetch_fyers(symbol)
                if data:
                    return data
                # Fallback to Yahoo Finance
                if "yfinance" in self.sources:
                    return self.fetch_yfinance(symbol)
            else:
                # International → Yahoo Finance
                if "yfinance" in self.sources:
                    return self.fetch_yfinance(symbol)
        
        elif source == "yfinance" and "yfinance" in self.sources:
            return self.fetch_yfinance(symbol)
        
        elif source == "fyers" and "fyers" in self.sources:
            return self.fetch_fyers(symbol)
        
        print(f"⚠️ No data source available for {symbol}")
        return None
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get latest live price"""
        data = self.fetch_market_data(symbol, source="auto")
        return data.get("latest_price") if data else None
    
    def _store_market_data(self, data: Dict):
        """Store market data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for record in data["data"]:
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
            print(f"💾 Stored {len(data['data'])} records for {data['symbol']}")
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            conn.close()
    
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
            print(f"✅ Added {quantity} shares of {symbol} at ${entry_price:.2f}")
        except Exception as e:
            print(f"Portfolio error: {e}")
        finally:
            conn.close()
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with current values"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM portfolio WHERE quantity > 0')
        positions = cursor.fetchall()
        conn.close()
        
        if not positions:
            return {"total_value": 0, "positions": []}
        
        summary = {
            "total_invested": 0,
            "total_current": 0,
            "total_pnl": 0,
            "positions": []
        }
        
        for pos in positions:
            symbol = pos[1]
            quantity = pos[2]
            entry_price = pos[3]
            
            # Get current price
            current_price = self.get_live_price(symbol)
            if not current_price:
                current_price = entry_price
            
            invested = quantity * entry_price
            current_value = quantity * current_price
            pnl = current_value - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0
            
            summary["positions"].append({
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "invested": invested,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
            
            summary["total_invested"] += invested
            summary["total_current"] += current_value
        
        summary["total_pnl"] = summary["total_current"] - summary["total_invested"]
        summary["total_pnl_pct"] = (summary["total_pnl"] / summary["total_invested"] * 100) if summary["total_invested"] > 0 else 0
        
        return summary

# ==================== TEST FUNCTION ====================

def test_pipeline():
    """Test the pipeline"""
    print("🧪 Testing Minimal Market Pipeline")
    print("="*50)
    
    # Initialize
    pipeline = MinimalMarketPipeline()
    
    # Test 1: Yahoo Finance
    print("\n1. Testing Yahoo Finance...")
    us_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in us_symbols:
        data = pipeline.fetch_market_data(symbol, source="yfinance")
        if data:
            print(f"   ✅ {symbol}: ${data['latest_price']:.2f} ({data['source']})")
        else:
            print(f"   ❌ {symbol}: Failed")
    
    # Test 2: Fyers (if available)
    print("\n2. Testing Fyers...")
    indian_symbols = ["NIFTY50-INDEX", "RELIANCE-EQ"]
    
    for symbol in indian_symbols:
        data = pipeline.fetch_market_data(symbol, source="fyers")
        if data:
            print(f"   ✅ {symbol}: ₹{data['latest_price']:.2f} ({data['source']})")
        else:
            print(f"   ❌ {symbol}: Failed or not authenticated")
    
    # Test 3: Portfolio
    print("\n3. Testing Portfolio...")
    # Add some test positions
    pipeline.add_portfolio_position("AAPL", 10, 150.25, "Test position")
    pipeline.add_portfolio_position("MSFT", 5, 300.50, "Another test")
    
    portfolio = pipeline.get_portfolio_summary()
    print(f"   Portfolio value: ${portfolio['total_current']:.2f}")
    print(f"   P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.1f}%)")
    
    print("\n" + "="*50)
    print("✅ Test completed!")

if __name__ == "__main__":
    test_pipeline()