# financial/financial_tools.py
"""
Professional Financial Tools for AI Agent
Integrates with ProfessionalMarketPipeline for institutional-grade analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class FinancialTools:
    """
    Professional financial analysis tools integrated with AI agent
    
    Features:
    - Multi-timeframe analysis
    - Volume profile interpretation
    - Technical indicators
    - Risk metrics
    - Natural language output for LLM
    """
    
    def __init__(self):
        """Initialize with professional pipeline"""
        self.pipeline = None
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Lazy load professional pipeline"""
        try:
            from data.professional_pipeline import ProfessionalMarketPipeline
            self.pipeline = ProfessionalMarketPipeline()
            print("✅ Professional pipeline connected to FinancialTools")
        except ImportError:
            print("⚠️ Professional pipeline not available, using fallback")
            self.pipeline = None
    
    # ========================================================================
    # CORE ANALYSIS METHODS (For AI Agent)
    # ========================================================================
    
    def analyze_stock_for_agent(self, symbol: str, detailed: bool = False) -> str:
        """
        Professional stock analysis for AI agent responses
        
        Args:
            symbol: Stock symbol
            detailed: If True, includes multi-timeframe + volume profile
        
        Returns:
            Natural language analysis for LLM consumption
        """
        try:
            if not self.pipeline:
                return self._fallback_analysis(symbol)
            
            # Get current price with quality validation
            data = self.pipeline.fetch_market_data(
                symbol=symbol,
                interval='1d',
                source='auto',
                validate=True
            )
            
            if not data:
                return f"❌ Could not fetch data for {symbol}"
            
            # Build analysis
            analysis_parts = []
            
            # 1. Basic price info with quality
            analysis_parts.append(f"📊 {symbol.upper()} ANALYSIS")
            analysis_parts.append(f"Current Price: ₹{data.latest_price:.2f}")
            analysis_parts.append(f"Data Quality: {data.quality.score:.1f}/5.0 ⭐")
            analysis_parts.append(f"Source: {data.source} ({data.latency_ms:.0f}ms)")
            
            # 2. Calculate basic metrics from recent data
            if data.data and len(data.data) >= 5:
                recent_closes = [bar['close'] for bar in data.data[-20:]]
                recent_volumes = [bar['volume'] for bar in data.data[-20:]]
                
                # Price change
                if len(recent_closes) > 1:
                    change_pct = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0]) * 100
                    analysis_parts.append(f"20-Day Change: {change_pct:+.2f}%")
                
                # Volatility
                if len(recent_closes) > 2:
                    volatility = (np.std(recent_closes) / np.mean(recent_closes)) * 100
                    analysis_parts.append(f"Volatility: {volatility:.1f}%")
                
                # Volume trend
                if len(recent_volumes) >= 10:
                    recent_avg_vol = np.mean(recent_volumes[-5:])
                    past_avg_vol = np.mean(recent_volumes[-10:-5])
                    vol_change = ((recent_avg_vol - past_avg_vol) / past_avg_vol) * 100 if past_avg_vol > 0 else 0
                    analysis_parts.append(f"Volume Trend: {vol_change:+.1f}%")
            
            # 3. Detailed analysis if requested
            if detailed:
                analysis_parts.append("\n🔍 DETAILED ANALYSIS:")
                
                # Multi-timeframe momentum
                mtf_summary = self._get_multi_timeframe_summary(symbol)
                if mtf_summary:
                    analysis_parts.append(mtf_summary)
                
                # Volume profile
                vp_summary = self._get_volume_profile_summary(symbol)
                if vp_summary:
                    analysis_parts.append(vp_summary)
            
            # 4. Data quality warnings
            if data.quality.warnings:
                analysis_parts.append(f"\n⚠️ Warnings: {', '.join(data.quality.warnings)}")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Analysis error for {symbol}: {str(e)[:100]}"
    
    def _fallback_analysis(self, symbol: str) -> str:
        """Fallback analysis when professional pipeline not available"""
        return f"""
    ❌ ANALYSIS UNAVAILABLE: {symbol}

    Professional pipeline not available.

    To enable analysis:
    1. Ensure professional_pipeline.py is installed
    2. Check financial/data/ directory structure
    3. Verify all dependencies are installed

    Current Status: Pipeline connection failed
    """
    
    # ========================================================================
    # MULTI-TIMEFRAME ANALYSIS
    # ========================================================================
    
    def _get_multi_timeframe_summary(self, symbol: str) -> str:
        """Get multi-timeframe momentum summary"""
        try:
            if not self.pipeline:
                return ""
            
            # Fetch multiple timeframes
            timeframes = ['1h', '1d']
            mtf_data = self.pipeline.fetch_multi_timeframe(
                symbol=symbol,
                timeframes=timeframes,
                source='auto'
            )
            
            if not mtf_data:
                return ""
            
            summary = ["Multi-Timeframe Momentum:"]
            
            for tf, data in mtf_data.items():
                if data.data and len(data.data) >= 2:
                    closes = [bar['close'] for bar in data.data]
                    if len(closes) >= 2:
                        change = ((closes[-1] - closes[0]) / closes[0]) * 100
                        trend = "📈 Bullish" if change > 0 else "📉 Bearish"
                        summary.append(f"  {tf}: {trend} ({change:+.2f}%)")
            
            return "\n".join(summary) if len(summary) > 1 else ""
            
        except Exception as e:
            return ""
    
    # ========================================================================
    # VOLUME PROFILE ANALYSIS
    # ========================================================================
    
    def _get_volume_profile_summary(self, symbol: str) -> str:
        """Get volume profile summary with key levels"""
        try:
            if not self.pipeline:
                return ""
            
            # Calculate volume profile
            vp = self.pipeline.calculate_volume_profile(symbol)
            
            if not vp:
                return ""
            
            summary = [
                "Volume Profile:",
                f"  POC (Point of Control): ₹{vp.poc:.2f}",
                f"  VAH (Value Area High): ₹{vp.vah:.2f}",
                f"  VAL (Value Area Low): ₹{vp.val:.2f}",
            ]
            
            # Interpret position relative to POC
            current_price = self.pipeline.fetch_market_data(symbol, interval='1d')
            if current_price and current_price.latest_price:
                price = current_price.latest_price
                if price > vp.vah:
                    summary.append("  🔴 Price above value area - potential resistance")
                elif price < vp.val:
                    summary.append("  🟢 Price below value area - potential support")
                else:
                    summary.append("  🟡 Price within value area - fair value zone")
            
            return "\n".join(summary)
            
        except Exception as e:
            return ""
    
    # ========================================================================
    # PORTFOLIO ANALYSIS
    # ========================================================================
    
    def get_portfolio_snapshot(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get professional portfolio snapshot with quality metrics
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dict with symbol -> data mapping
        """
        snapshot = {}
        
        if not self.pipeline:
            # Fallback to basic method
            return self._fallback_portfolio(symbols)
        
        for symbol in symbols:
            try:
                data = self.pipeline.fetch_market_data(
                    symbol=symbol,
                    interval='1d',
                    source='auto',
                    validate=True
                )
                
                if data:
                    snapshot[symbol] = {
                        'price': data.latest_price,
                        'source': data.source,
                        'quality': data.quality.score,
                        'latency_ms': data.latency_ms,
                        'timestamp': data.timestamp
                    }
            except:
                continue
        
        return snapshot
    
    def _fallback_portfolio(self, symbols: List[str]) -> Dict[str, Any]:
        """Fallback portfolio snapshot"""
        return {
            symbol: {
                'price': None,
                'source': 'unavailable',
                'error': 'Professional pipeline not available',
                'timestamp': datetime.now().isoformat()
            } 
            for symbol in symbols
        }
   
    
    # ========================================================================
    # FORMATTED OUTPUT FOR LLM
    # ========================================================================
    
    def format_market_insight(self, symbol: str, data: Dict) -> str:
        """
        Format market data for LLM analysis
        
        Args:
            symbol: Stock symbol
            data: Market data dict (from pipeline or enhanced_get_market_data)
        
        Returns:
            Formatted insight string
        """
        if not data or data.get('status') == 'no_data':
            return "❌ No market data available"
        
        # Handle both pipeline response and legacy response
        if hasattr(data, 'latest_price'):
            # ProfessionalMarketPipeline response
            insight = f"""
📊 MARKET INSIGHT: {symbol}
────────────────────
Price: ₹{data.latest_price:.2f}
Quality: {data.quality.score:.1f}/5.0 ⭐
Source: {data.source}
Latency: {data.latency_ms:.0f}ms
Records: {data.record_count}
Time: {data.timestamp}
"""
            if data.quality.warnings:
                insight += f"\n⚠️  {', '.join(data.quality.warnings)}"
            
            return insight
        
        else:
            # Legacy response format
            price = data.get('price', 0)
            open_price = data.get('open', 0)
            high = data.get('high', 0)
            low = data.get('low', 0)
            volume = data.get('volume', 0)
            
            insight = f"""
📊 MARKET INSIGHT: {symbol}
────────────────────
Price: ₹{price:.2f}
Open: ₹{open_price:.2f}
High: ₹{high:.2f}  
Low: ₹{low:.2f}
Volume: {volume:,}
Source: {data.get('source', 'unknown')}
"""
            return insight
    
    # ========================================================================
    # TECHNICAL INDICATORS (For Advanced Analysis)
    # ========================================================================
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        try:
            if not self.pipeline:
                return None
            
            data = self.pipeline.fetch_market_data(symbol, interval='1d')
            
            if not data or not data.data or len(data.data) < period + 1:
                return None
            
            closes = [bar['close'] for bar in data.data]
            
            # Calculate price changes
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Average gains and losses
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            return None
    
    def calculate_moving_averages(self, symbol: str, periods: List[int] = [20, 50, 200]) -> Dict[int, float]:
        """Calculate multiple moving averages"""
        try:
            if not self.pipeline:
                return {}
            
            data = self.pipeline.fetch_market_data(symbol, interval='1d')
            
            if not data or not data.data:
                return {}
            
            closes = [bar['close'] for bar in data.data]
            
            mas = {}
            for period in periods:
                if len(closes) >= period:
                    mas[period] = np.mean(closes[-period:])
            
            return mas
            
        except Exception as e:
            return {}
    
    # ========================================================================
    # RISK METRICS
    # ========================================================================
    
    def calculate_volatility(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate historical volatility (annualized)"""
        try:
            if not self.pipeline:
                return None
            
            data = self.pipeline.fetch_market_data(symbol, interval='1d')
            
            if not data or not data.data or len(data.data) < period:
                return None
            
            closes = [bar['close'] for bar in data.data[-period:]]
            returns = np.diff(np.log(closes))
            
            # Annualize (252 trading days)
            volatility = np.std(returns) * np.sqrt(252) * 100
            
            return volatility
            
        except Exception as e:
            return None
    
    def calculate_sharpe_ratio(self, symbol: str, risk_free_rate: float = 0.05) -> Optional[float]:
        """Calculate Sharpe ratio"""
        try:
            if not self.pipeline:
                return None
            
            data = self.pipeline.fetch_market_data(symbol, interval='1d')
            
            if not data or not data.data or len(data.data) < 30:
                return None
            
            closes = [bar['close'] for bar in data.data]
            returns = np.diff(closes) / closes[:-1]
            
            # Annualize
            avg_return = np.mean(returns) * 252
            std_return = np.std(returns) * np.sqrt(252)
            
            if std_return == 0:
                return None
            
            sharpe = (avg_return - risk_free_rate) / std_return
            
            return sharpe
            
        except Exception as e:
            return None
    
    # ========================================================================
    # COMPREHENSIVE REPORT (For Detailed Analysis)
    # ========================================================================
    
    def generate_comprehensive_report(self, symbol: str) -> str:
        """
        Generate comprehensive analysis report for AI agent
        Combines all analysis methods
        """
        try:
            report_parts = []
            
            # Header
            report_parts.append(f"📈 COMPREHENSIVE ANALYSIS: {symbol.upper()}")
            report_parts.append("="*50)
            
            # 1. Basic analysis
            basic = self.analyze_stock_for_agent(symbol, detailed=True)
            report_parts.append(basic)
            
            # 2. Technical indicators
            report_parts.append("\n📊 TECHNICAL INDICATORS:")
            
            rsi = self.calculate_rsi(symbol)
            if rsi:
                rsi_signal = "Overbought 🔴" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral 🟡"
                report_parts.append(f"  RSI(14): {rsi:.1f} - {rsi_signal}")
            
            mas = self.calculate_moving_averages(symbol)
            if mas:
                report_parts.append("  Moving Averages:")
                for period, value in mas.items():
                    report_parts.append(f"    MA{period}: ₹{value:.2f}")
            
            # 3. Risk metrics
            report_parts.append("\n⚠️ RISK METRICS:")
            
            volatility = self.calculate_volatility(symbol)
            if volatility:
                risk_level = "High 🔴" if volatility > 40 else "Medium 🟡" if volatility > 20 else "Low 🟢"
                report_parts.append(f"  Volatility: {volatility:.1f}% - {risk_level}")
            
            sharpe = self.calculate_sharpe_ratio(symbol)
            if sharpe:
                sharpe_rating = "Excellent 🟢" if sharpe > 2 else "Good 🟡" if sharpe > 1 else "Poor 🔴"
                report_parts.append(f"  Sharpe Ratio: {sharpe:.2f} - {sharpe_rating}")
            
            # 4. Data quality summary
            if self.pipeline:
                quality_report = self.pipeline.get_data_quality_report(symbol)
                if quality_report:
                    report_parts.append("\n✅ DATA QUALITY:")
                    for source, metrics in quality_report.items():
                        report_parts.append(f"  {source}: {metrics['avg_quality']}")
            
            return "\n".join(report_parts)
            
        except Exception as e:
            return f"❌ Report generation error: {str(e)[:100]}"


# ========================================================================
# STANDALONE TESTING
# ========================================================================

def test_financial_tools():
    """Test financial tools with professional pipeline"""
    print("🧪 Testing Financial Tools")
    print("="*60)
    
    tools = FinancialTools()
    
    # Test 1: Basic analysis
    print("\n1. Basic Analysis:")
    analysis = tools.analyze_stock_for_agent("RELIANCE")
    print(analysis)
    
    # Test 2: Detailed analysis
    print("\n2. Detailed Analysis:")
    detailed = tools.analyze_stock_for_agent("NIFTY", detailed=True)
    print(detailed)
    
    # Test 3: Portfolio snapshot
    print("\n3. Portfolio Snapshot:")
    portfolio = tools.get_portfolio_snapshot(["RELIANCE", "TCS", "INFY"])
    for symbol, data in portfolio.items():
        print(f"  {symbol}: ₹{data['price']:.2f} (Q: {data.get('quality', 'N/A')})")
    
    # Test 4: Comprehensive report
    print("\n4. Comprehensive Report:")
    report = tools.generate_comprehensive_report("TCS")
    print(report)
    
    print("\n" + "="*60)
    print("✅ Financial Tools Test Complete!")


if __name__ == "__main__":
    test_financial_tools()