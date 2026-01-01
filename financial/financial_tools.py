# financial/financial_tools.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FinancialTools:
    """Financial analysis tools that integrate with your AI agent"""
    
    @staticmethod
    def analyze_stock_for_agent(symbol, data_points=20):
        """
        Simple stock analysis for AI agent responses
        Returns: Analysis text for LLM consumption
        """
        try:
            # Use the existing market data function
            from main import enhanced_get_market_data  # Import from your main.py
            
            # Get recent data
            data = enhanced_get_market_data(symbol, timeframe="5d", interval="1d")
            
            if data.get('status') != 'ok':
                return f"Could not fetch data for {symbol}"
            
            # Simple analysis
            price = data.get('price', 0)
            
            # Check if it's a list or dict
            if isinstance(data.get('data'), list) and len(data['data']) > 1:
                # Multiple data points available
                prices = [d.get('close', price) for d in data['data'][-data_points:]]
                if len(prices) > 1:
                    change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
                    volatility = np.std(prices) / np.mean(prices) * 100 if len(prices) > 2 else 0
                    
                    analysis = f"""
{symbol} Analysis:
- Current Price: ₹{price:.2f}
- {len(prices)}-period Change: {change_pct:+.2f}%
- Relative Volatility: {volatility:.1f}%
- Data Source: {data.get('source', 'unknown')}
"""
                    return analysis
            
            # Basic single point analysis
            return f"{symbol}: ₹{price:.2f} (Source: {data.get('source', 'unknown')})"
            
        except Exception as e:
            return f"Analysis error for {symbol}: {str(e)[:100]}"
    
    @staticmethod
    def get_portfolio_snapshot(symbols):
        """Get snapshot of multiple symbols"""
        snapshot = {}
        
        for symbol in symbols:
            try:
                data = enhanced_get_market_data(symbol, timeframe="1d")
                if data.get('status') == 'ok':
                    snapshot[symbol] = {
                        'price': data.get('price', 0),
                        'source': data.get('source', 'unknown'),
                        'timestamp': data.get('timestamp', datetime.now().isoformat())
                    }
            except:
                continue
        
        return snapshot
    
    @staticmethod
    def format_market_insight(symbol, data):
        """Format market data for LLM analysis"""
        if not data or data.get('status') != 'ok':
            return "No market data available"
        
        insight = f"""
📊 MARKET INSIGHT: {symbol}
────────────────────
Price: ₹{data.get('price', 0):.2f}
Open: ₹{data.get('open', 0):.2f}
High: ₹{data.get('high', 0):.2f}  
Low: ₹{data.get('low', 0):.2f}
Volume: {data.get('volume', 0):,}
Source: {data.get('source', 'unknown')}
Time: {data.get('timestamp', 'N/A')}
"""
        return insight