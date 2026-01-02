# 🤖 AI AGENT CONTEXT - DUAL-MODE PERSONAL & TRADING COMPANION

## 🎯 AGENT IDENTITY
- **Name**: Umar's AI Agent
- **Type**: Dual-Mode Personal & Trading Companion
- **Owner**: Umar Farooqi (User + Professional Trader)
- **Core**: Single agent with personal/trading modes
- **Constitution**: Follows AGENT_CONSTITUTION in main.py

## 🔄 DUAL MODE SYSTEM

### MODE: "personal"
- **For**: Personal life, relationships, memory, daily tasks
- **Memory**: core_identity.json (wife, friends, preferences)
- **Examples**: "My wife Sarah likes coffee", "Remember I prefer dark mode"
- **Entities**: wife, husband, friend, family, self

### MODE: "trading"  
- **For**: Professional trading, market analysis, Indian markets
- **Memory**: state_memory.json (beliefs, biases, strategies)
- **Examples**: "I'm bullish on NIFTY", "Analyze RELIANCE"
- **Focus**: NSE/BSE, Fyers/Upstox, real-time data

### AUTO-SWITCHING:
- Detects keywords, switches mode automatically
- Maintains separate memory for each domain
- Single conversation thread, dual contexts

## 📍 CURRENT STATE (What Actually Exists)

### ✅ WORKING PERSONAL FEATURES:
1. **Multi-tier Memory**: working_memory → core_identity promotion
2. **Auto-learning**: From conversation without explicit "remember"
3. **Relationship tracking**: wife, friends, family (PERSON_ENTITIES)
4. **Conflict resolution**: Handles "wife is Sarah" vs "wife is Priya"
5. **Confidence decay**: Facts fade if not reinforced
6. **User confirmation**: Asks before saving conflicting info

### ✅ WORKING TRADING FEATURES:
1. **Market data**: yfinance + MinimalMarketPipeline
2. **Indian symbols**: NSE/BSE with .NS suffix
3. **Broker auth**: Fyers + Upstox (authentication flows)
4. **Basic analysis**: financial_tools.py (price, change %)
5. **Market query detection**: "What's NIFTY at?" → auto-fetch
6. **Trading journal**: auto-logs trading conversations

### ✅ CORE INFRASTRUCTURE:
1. **Streamlit UI**: Web interface with sidebar mode selector
2. **LLM Integration**: RouteLLM with constitution
3. **Intent recognition**: 7+ intent types
4. **File processing**: Upload PDFs/images for analysis
5. **Conversation history**: Maintains context

## 🎯 WHAT YOU WANT (Professional Trading Upgrade)

### NEED TO TRANSFORM: Trading Mode from BASIC → PROFESSIONAL
**Current trading**: Basic price checks, delayed data  
**Target trading**: Institutional-grade Indian market companion

### PRIORITY 1: REAL-TIME TRADING INFRASTRUCTURE
- [ ] WebSocket data (Fyers/Upstox) for live prices
- [ ] Options chain analyzer (Indian expiries, Greeks)
- [ ] Multi-timeframe analysis (1m to 1D)

### PRIORITY 2: PROFESSIONAL ANALYTICS
- [ ] Advanced technical indicators (custom + standard)
- [ ] Risk management (position sizing, VaR)
- [ ] Portfolio analytics (correlation, concentration)

### PRIORITY 3: TRADING AUTOMATION
- [ ] Backtesting engine (your strategies)
- [ ] Signal generation + alerts
- [ ] Trade journal 2.0 (PnL tracking, lessons)

### PRIORITY 4: PERSONAL-TRADING INTEGRATION
- [ ] Mood detection affecting trading decisions
- [ ] Personal schedule → trading availability
- [ ] Risk tolerance based on personal context

## 🔧 CRITICAL: PRESERVE PERSONAL MODE

### DO NOT BREAK PERSONAL FEATURES:
1. **Memory system** (core_identity.json) - Stores wife/family info
2. **Auto-learning** - From casual conversation
3. **Confirmation system** - "Yes, that's correct" handling
4. **PERSON_ENTITIES** - wife, friend, etc. recognition

### PERSONAL MODE IS WORKING WELL:
- User: "My wife Sarah likes coffee" → saves to working memory
- User: "What's my wife's name?" → queries core_identity
- User: "Yes, that's correct" → promotes to core
- This all works! Don't break it.

## 🏗️ ARCHITECTURE RULE

### ADDITIVE, NOT DESTRUCTIVE:
- Keep all personal features working
- Add trading features alongside
- Two modes share same agent brain
- Memory separated by domain

### FILE STRUCTURE (Current):