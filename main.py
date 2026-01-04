# ==============================
# PERFORMANCE OPTIMIZATION
# ==============================
import sys
import os
import json
import re
import sqlite3
from datetime import datetime, timedelta
import time
from typing import Any, Optional, Callable
import hashlib
import uuid
import pandas as pd 
sys.dont_write_bytecode = True

# ==============================
# STREAMLIT & ENV
# ==============================
import streamlit as st
from dotenv import load_dotenv
import requests

# Load environment variables FIRST
load_dotenv()

# ==============================
# SESSION MANAGEMENT
# ==============================
def ensure_session_initialized():
    """Initialize session tracking for optimization"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_started_at = datetime.now().isoformat()
        st.session_state.constitution_sent = False
        st.session_state.llm_calls_count = 0
        print(f"🆕 New session started: {st.session_state.session_id}")

# NOW your existing code starts:
CURRENT_MODE = "personal"  # personal | trading
UI_STATUS = "Online"  # Online | Rate-limited | Offline

# LAZY LOADING: Initialize flags but don't import until needed
ENHANCED_MEMORY_AVAILABLE = None  # Will be set when first accessed
vector_memory = None
personality_engine = None
pattern_recognizer = None
_yfinance_module = None  # For lazy loading yfinance

# LAZY LOADING: Financial modules will be loaded on demand
PROFESSIONAL_PIPELINE_AVAILABLE = None
professional_pipeline = None
MARKET_PIPELINE_AVAILABLE = None
market_pipeline = None
MARKET_TOOLS_AVAILABLE = None
financial_tools = None
UPSTOX_AVAILABLE = None
upstox_auth_flow = None
check_upstox_auth = lambda: False  # Default fallback
UpstoxAuth = None

# Keep the test_fyers_integration function as is
def test_fyers_integration():
    """Test Fyers integration separately"""
    print("🔐 Testing Fyers integration...")
    
    try:
        from financial.auth.auth_helper import setup_fyers_auth
        token = setup_fyers_auth()
        if token:
            print(f"✅ Authentication successful")
            
            # Lazy load market pipeline if needed
            lazy_load_market_pipeline()
            
            # Test Fyers data fetch
            if MARKET_PIPELINE_AVAILABLE and market_pipeline:
                print("📊 Testing Fyers data fetch...")
                # Check if fetch_fyers method exists
                if hasattr(market_pipeline, 'fetch_fyers'):
                    data = market_pipeline.fetch_fyers("NSE:NIFTY50-INDEX")
                    if data:
                        print(f"✅ NIFTY data via Fyers: ₹{data.get('latest_price', 0):.2f}")
                    else:
                        print("⚠️ Fyers data fetch failed")
                else:
                    print("ℹ️ market_pipeline doesn't have fetch_fyers method")
        else:
            print("ℹ️ Fyers not authenticated. Using Yahoo Finance for Indian data.")
    except ImportError as e:
        print(f"⚠️ Fyers auth import error: {e}")

# Add this lazy loader function after test_fyers_integration
def lazy_load_financial_tools():
    """Lazy load financial tools when needed"""
    global MARKET_TOOLS_AVAILABLE, financial_tools, UPSTOX_AVAILABLE, upstox_auth_flow, check_upstox_auth, UpstoxAuth
    
    # Load financial tools if not already loaded
    if MARKET_TOOLS_AVAILABLE is None:
        try:
            from financial.financial_tools import FinancialTools
            financial_tools = FinancialTools()
            MARKET_TOOLS_AVAILABLE = True
            print("✅ Lazy loaded financial tools")
        except ImportError as e:
            MARKET_TOOLS_AVAILABLE = False
            financial_tools = None
            print(f"⚠️ Financial tools import failed: {e}")
    
    # Load Upstox if not already loaded
    if UPSTOX_AVAILABLE is None:
        try:
            from financial.auth.upstox_auth import upstox_auth_flow as uaf, check_upstox_auth as cua, UpstoxAuth as UA
            upstox_auth_flow = uaf
            check_upstox_auth = cua
            UpstoxAuth = UA
            UPSTOX_AVAILABLE = True
            print("✅ Lazy loaded Upstox auth tools")
        except ImportError as e:
            UPSTOX_AVAILABLE = False
            upstox_auth_flow = None
            check_upstox_auth = lambda: False
            UpstoxAuth = None
            print(f"⚠️ Upstox auth not available: {e}")

# Add this pipeline lazy loader function
def lazy_load_market_pipeline():
    """Lazy load market pipelines when needed"""
    global PROFESSIONAL_PIPELINE_AVAILABLE, professional_pipeline, MARKET_PIPELINE_AVAILABLE, market_pipeline
    
    # Try professional pipeline first
    if PROFESSIONAL_PIPELINE_AVAILABLE is None:
        try:
            from financial.data.professional_pipeline import ProfessionalMarketPipeline
            professional_pipeline = ProfessionalMarketPipeline(db_path="financial/data/professional.db")
            PROFESSIONAL_PIPELINE_AVAILABLE = True
            print("✅ Lazy loaded professional pipeline")
        except ImportError as e:
            PROFESSIONAL_PIPELINE_AVAILABLE = False
            print(f"⚠️ Professional pipeline import failed: {e}")
    
    # Fallback to minimal pipeline
    if MARKET_PIPELINE_AVAILABLE is None:
        try:
            from financial.data.minimal_pipeline import MinimalMarketPipeline
            market_pipeline = MinimalMarketPipeline()
            MARKET_PIPELINE_AVAILABLE = True
            print("✅ Lazy loaded professional pipeline")
        except ImportError as e:
            MARKET_PIPELINE_AVAILABLE = False
            # Create dummy
            class DummyMarketPipeline:
                def __init__(self):
                    self.sources = ["dummy"]
                def fetch_market_data(self, *args, **kwargs):
                    return None
            market_pipeline = DummyMarketPipeline()
            print(f"⚠️ Market pipeline import failed, using dummy: {e}")
        
# Constants moved from original position
IDENTITY_CONFLICT_WINDOW = timedelta(days=2)
CONFIDENCE_DECAY_PER_DAY = 0.05
CONFIDENCE_MIN_THRESHOLD = 0.3
SOFT_DELETE_AFTER_DAYS = 14
PENDING_IDENTITY_CONFIRMATION = None

# ==============================
# ABSOLUTE PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

# ==============================
# MEMORY SYSTEM - SIMPLIFIED VERSION
# ==============================

def load_json(path, default):
    """Safely load a JSON file."""
    # Make path absolute if it's relative
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default

def save_json(path, data):
    """Safely save a JSON file."""
    # Make path absolute if it's relative
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def apply_memory_action(action):
    """
    MEMORY WRITE PRIMITIVE (LOW-LEVEL)

    IMPORTANT DESIGN CONTRACT:
    --------------------------
    This function is a *dumb write primitive*.
    It MUST NOT decide:
      - cardinality (single vs multiple)
      - conflicts
      - overwrites
      - promotions

    All semantic decisions (cardinality, conflict resolution,
    duplication, user confirmation) MUST be handled
    BEFORE calling this function.

    apply_memory_action() assumes:
      - the caller has already validated correctness
      - the write is intentional and authorized
      - overwriting rules have already been enforced

    This separation is intentional to preserve:
      - predictability
      - auditability
      - debuggability

    If cardinality logic appears here, it is a BUG.
    """
    memory_type = action.get("type", "")
    domain = action.get("domain", "personal")
    
    if memory_type == "ADD_FACT":
        # NOTE: Cardinality/conflict checks are intentionally NOT done here.
        # They MUST be enforced by the caller before invoking this write.
        if domain == "personal":
            file_path = os.path.join(MEMORY_DIR, "core_identity.json")
        elif domain == "trading":
            file_path = os.path.join(MEMORY_DIR, "state_memory.json")
        elif domain == "diary":
            file_path = os.path.join(MEMORY_DIR, "diary.json")
        else:
            file_path = os.path.join(MEMORY_DIR, "state_memory.json")
        
        # Load existing data
        data = load_json(file_path, {"facts": []})
        
        # Check for conflicts (simplified)
        for fact in data.get("facts", []):
            if (fact.get("entity") == action.get("entity") and 
                fact.get("attribute") == action.get("attribute")):
                return {"conflict": True, "existing_value": fact.get("value")}
        
        # Add new fact
        new_fact = {
            "entity": action.get("entity"),
            "attribute": action.get("attribute"),
            "value": action.get("value"),
            "confidence": action.get("confidence", 0.8),
            "timestamp": datetime.now().isoformat(),
            "source": action.get("source", "user")
        }
        
        data.setdefault("facts", []).append(new_fact)
        save_json(file_path, data)
        return {"applied": True}
    
    return {"applied": False}

def query_fact(domain, entity, attribute, memory_data=None):
    """Query a fact from memory."""
    if memory_data is None:
        if domain == "personal":
            memory_data = load_json(os.path.join(MEMORY_DIR, "core_identity.json"), {"facts": []})
        elif domain == "trading":
            memory_data = load_json(os.path.join(MEMORY_DIR, "state_memory.json"), {"states": []})
        else:
            memory_data = {"facts": []}
    
    # Handle both old and new memory structures
    facts = memory_data.get("facts", []) or memory_data.get("states", [])
    
    for fact in facts:
        if (fact.get("entity") == entity and 
            fact.get("attribute") == attribute):
            return fact.get("value")
    
    return None

def initialize_all_memory_files():
    """Initialize all memory files if they don't exist."""
    memory_files = {
        os.path.join(MEMORY_DIR, "working_memory.json"): {"observations": []},
        os.path.join(MEMORY_DIR, "core_identity.json"): {"facts": []},
        os.path.join(MEMORY_DIR, "state_memory.json"): {"states": []},
        os.path.join(MEMORY_DIR, "bias.json"): {
            "current": None,
            "based_on": None,
            "confidence": None,
            "invalidated_if": None,
            "last_invalidated_reason": None,
            "history": []
        },
        os.path.join(MEMORY_DIR, "trading_journal.json"): {"entries": []},
        os.path.join(MEMORY_DIR, "promotion_audit.json"): {"events": []},
        os.path.join(MEMORY_DIR, "diary.json"): {"entries": []}
    }
    
    for file_path, default_data in memory_files.items():
        if not os.path.exists(file_path):
            save_json(file_path, default_data)
            print(f"Created missing memory file: {file_path}")

def load_identity_memory():
    """Load identity memory from file."""
    return load_json(os.path.join(MEMORY_DIR, "core_identity.json"), {"facts": []})

def promote_to_preferences(working_memory, core_identity):
    """
    Simplified preference promotion.
    In a real implementation, this would have more logic.
    """
    # This is a placeholder - in reality, this would analyze working memory
    # and promote certain observations to core identity
    return False

def process_file(file_data):
    """
    Simplified file processor.
    Returns processed file content.
    """
    # In a real implementation, this would process different file types
    if hasattr(file_data, 'read'):
        content = file_data.read()
        if hasattr(content, 'decode'):
            content = content.decode('utf-8')
        return {"content": content, "type": "text"}
    return {"content": str(file_data), "type": "unknown"}

def format_file_for_llm(processed_data, user_input):
    """
    Format file content for LLM consumption.
    """
    if processed_data.get("type") == "text":
        formatted_input = f"File content: {processed_data['content'][:1000]}...\n\nUser query: {user_input}"
        return formatted_input, None
    return user_input, None

# ==============================
# INITIALIZE MEMORY
# ==============================
os.makedirs(MEMORY_DIR, exist_ok=True)
initialize_all_memory_files()

working_memory = load_json(
    os.path.join(MEMORY_DIR, "working_memory.json"),
    {"observations": []}
)

core_identity = load_json(
    os.path.join(MEMORY_DIR, "core_identity.json"),
    {"facts": []}
)

state_memory = load_json(
    os.path.join(MEMORY_DIR, "state_memory.json"),
    {"states": []}
)

bias_memory = load_json(
    os.path.join(MEMORY_DIR, "bias.json"),
    {
        "current": None,
        "based_on": None,
        "confidence": None,
        "invalidated_if": None,
        "last_invalidated_reason": None,
        "history": []
    }
)

# ==============================
# PERSON ENTITIES
# ==============================
PERSON_ENTITIES = {
    "self", "wife", "husband", "partner",
    "father", "mother", "brother", "sister",
    "son", "daughter", "friend", "best_friend"
}

# ==============================
# DATA ROUTER & SYMBOL INTELLIGENCE (SIMPLIFIED)
# ==============================
def get_market_data(symbol, timeframe="1d"):
    """
    Enhanced market data fetcher using your working MinimalMarketPipeline
    Returns: (data_dict, source, status)
    """
    # LAZY LOAD: Initialize financial modules if not already loaded
    global MARKET_PIPELINE_AVAILABLE, market_pipeline, _yfinance_module
    
    # If enhanced pipeline is available, use it
    if MARKET_PIPELINE_AVAILABLE is None:
        # Lazy load market pipeline
        try:
            from financial.data.minimal_pipeline import MinimalMarketPipeline
            market_pipeline = MinimalMarketPipeline()
            MARKET_PIPELINE_AVAILABLE = True
            print("✅ Lazy loaded market pipeline")
        except ImportError as e:
            MARKET_PIPELINE_AVAILABLE = False
            # Create dummy
            class DummyMarketPipeline:
                def __init__(self):
                    self.sources = ["dummy"]
                def fetch_market_data(self, *args, **kwargs):
                    return None
            market_pipeline = DummyMarketPipeline()
    
    if MARKET_PIPELINE_AVAILABLE and market_pipeline:
        try:
            # Call pipeline WITHOUT interval parameter
            data = market_pipeline.fetch_market_data(symbol, source="auto")
            
            if not data:
                return None, "pipeline", "no_data"
            
            # Extract latest price from pipeline response
            latest_price = data.get("latest_price", 0)
            data_records = data.get("data", [])
            
            if not data_records:
                return None, data.get("source", "unknown"), "no_records"
            
            # Get latest record
            latest_record = data_records[-1]
            
            return {
                'symbol': symbol,
                'price': float(latest_price),
                'open': float(latest_record.get('open', latest_price)),
                'high': float(latest_record.get('high', latest_price)),
                'low': float(latest_record.get('low', latest_price)),
                'volume': int(latest_record.get('volume', 0)),
                'timestamp': latest_record.get('timestamp', datetime.now().isoformat()),
                'source': data.get('source', 'unknown'),
                'interval': data.get('interval', '1d')
            }, data.get('source', 'unknown'), "ok"
            
        except Exception as e:
            print(f"Enhanced pipeline error for {symbol}: {e}")
            # Fall through to simple method
    
    # FALLBACK: Simple yfinance method (always works) with lazy loading
    try:
        # LAZY LOAD yfinance
        global _yfinance_module
        if _yfinance_module is None:
            import yfinance as yf
            _yfinance_module = yf
            print("✅ Lazy loaded yfinance")
        
        # Map timeframe to yfinance parameters
        period_map = {
            "1min": "1d",
            "5min": "5d",
            "15min": "5d",
            "30min": "5d",
            "1h": "5d",
            "1d": "5d",
            "1w": "1mo",
            "1M": "3mo"
        }
        
        interval_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo"
        }
        
        period = period_map.get(timeframe, "5d")
        interval = interval_map.get(timeframe, "1d")
        
        print(f"📊 Fallback: Fetching {symbol} ({period}/{interval}) via yfinance...")
        
        ticker = _yfinance_module.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, "yfinance", "no_data"
        
        # Get latest data
        latest = df.iloc[-1]
        
        return {
            'symbol': symbol,
            'price': float(latest.get('Close', 0)),
            'open': float(latest.get('Open', 0)),
            'high': float(latest.get('High', 0)),
            'low': float(latest.get('Low', 0)),
            'volume': int(latest.get('Volume', 0)),
            'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name),
            'source': 'yfinance',
            'interval': interval
        }, "yfinance", "ok"
        
    except Exception as e:
        print(f"Market data fallback error: {e}")
        return None, "error", f"error: {str(e)[:100]}"
    
def resolve_instrument(user_input):
    """Simplified instrument resolver."""
    # In reality, this would parse the input to find instrument symbols
    return {"symbol": "NIFTY", "type": "index"}

# ==============================
# CONSTITUTION & MEMORY POLICY
# ==============================
AGENT_CONSTITUTION = """
You are an AI Agent operating under the following Constitution.
You must follow it strictly.

====================
AGENT CONSTITUTION — v1.0 (COMPRESSED)
Owner: Umar Farooqi
Status: Stable

OVERRIDING RULE
Correctness, clarity, and stability override fluency, speed, or creativity.

1. PURPOSE
The Agent is a calm, structured thinking partner.
It reduces cognitive load and improves clarity and decision quality.
It is not a filler chatbot, opinionated authority, or code-first tool.

2. CORE PRINCIPLES
- Stability over novelty: do not change tools or approaches without clear gain.
- Plain-text first: behavior and logic must be representable in text.
- Surgical changes only: preserve working systems.
- Clarity over verbosity: be concise and structured.

3. BEHAVIOR
- Calm, grounded tone; structured output.
- Ask questions only when necessary.
- Make reasonable assumptions and state them.
- When errors occur: explain what failed, why, and the minimal fix.

4. CONVERSATION
- Natural conversation is allowed as an output layer only.
- Conversation does not override memory, role modes, or control rules.

5. ROLES (IMPLICIT)
- Analyst: neutral, evidence-based evaluation.
- Builder: stepwise, modular system design.
- Explainer: simple language, minimal jargon.

6. MEMORY
- Memory exists to improve consistency, not to store everything.
- Never overwrite identity silently.
- Memory writes require confirmation or policy approval.
- The Agent must never claim to perform memory writes itself.

7. MODEL & TOOLS
- Model-agnostic by design.
- Model swaps must not change behavior.
- The constitution is invariant.

8. FAILURE
- State uncertainty explicitly.
- Never hallucinate.
- Reduce scope when overloaded.

9. USER AUTHORITY
- User commands override all prior beliefs.
- No arguing, no silent behavior changes.

10. EVOLUTION
- Changes must be intentional, versioned, and documented.
- No silent drift.

FINAL DIRECTIVE
The Agent exists to work, not to impress.
Consistency > Cleverness
Clarity > Complexity

====================

Operational Rules:
- The Constitution overrides all other instructions.
- Default to stability, clarity, and surgical changes.
- Conversation is allowed as defined in Section 3.4.
"""

# ==============================
# FINANCIAL INTELLIGENCE PROTOCOL
# ==============================
FINANCIAL_INTELLIGENCE = """
FINANCIAL INTELLIGENCE — STRICT

1. TOOLS
- Financial tools are authoritative and always used.
- The model must not claim lack of access to market data.
- If tools fail, report unavailability honestly.

2. CONTEXT
- Follow-ups inherit the last discussed symbol automatically.
- Comparisons include the previous symbol unless stated otherwise.

3. DEFAULT ROUTING
- “Analyze [STOCK]” → comprehensive report
- “RSI / MA / Volatility” → indicator on last symbol
- “Price of [STOCK]” → direct price fetch

4. OUTPUT
- Be direct and concise.
- Include interpretation when relevant.
- Do not apologize for data access.

5. PROHIBITIONS
- Do not say “no access to live data”.
- Do not invent prices or indicators.

6. CLARIFICATION
- If intent or symbol is unclear, ask briefly.
"""

MEMORY_POLICY = """
MEMORY POLICY — v2.0 (COMPRESSED)

PURPOSE
Memory improves long-term consistency and personalization.
It is minimal, factual, reversible, and transparent.

MEMORY TIERS
1. Working Memory: temporary observations and candidates.
2. Core Identity: verified, stable personal facts.
3. State Memory: session or temporal beliefs.
4. Specialized: trading journal, bias memory, audit logs.

AUTO-LEARNING
Candidates may be created only from:
- personal factual statements
- repeated, consistent mentions
Never from:
- questions
- casual chat
- ambiguous statements

PROMOTION RULES
Promotion requires ANY:
- confidence ≥ 0.75
- repeated mentions ≥ 2
- explicit confirmation

Promotion is blocked by ANY:
- active conflict window
- confidence decay < threshold
- inactivity

CARDINALITY
- Single-value roles: must not be overwritten silently.
- Multi-value roles: append-only, no duplicates.
- Conflicts require explicit user resolution.

CONFIRMATION
Triggered when:
- pending candidates exist
- conflicts are detected
- blocked promotions remain

DECAY
- Confidence decays daily.
- Low-confidence or inactive items are inactivated.
- All decisions are audited.

USER AUTHORITY
- Explicit commands override all inference.
- Queries are read-only.
- Casual chat triggers no memory ops.

PROHIBITIONS
The Agent must not:
- store sensitive data without instruction
- modify or delete identity silently
- hallucinate memory content

TRANSPARENCY
Users may request:
- what is known
- why something was not saved
- audit explanations

FINAL RULE
Memory must remain minimal, factual, reversible, and user-controlled.

"""
# ==============================
# OPTIMIZATION CONSTANTS
# ==============================
SYSTEM_CORE = """You are a calm, structured AI assistant focused on reducing cognitive load and improving clarity."""
FULL_CONSTITUTION = AGENT_CONSTITUTION.strip()
FULL_MEMORY_POLICY = MEMORY_POLICY.strip()
FULL_FINANCIAL_INTELLIGENCE = FINANCIAL_INTELLIGENCE.strip()

from collections import defaultdict

# Remove the problematic constitution update line since it's not needed
# AGENT_CONSTITUTION = AGENT_CONSTITUTION.replace("MEMORY POLICY", "MEMORY POLICY V2.0")

# ================================
# OPTIMIZATION HELPER FUNCTIONS
# ================================
def should_send_full_constitution(intent_data: dict, first_call_of_session: bool) -> bool:
    """Determine if we need to send the full constitution"""
    if first_call_of_session:
        return True
    
    # Only send full constitution for memory/identity operations
    if intent_data.get("primary_intent") == "memory":
        return True
    
    return False

def should_send_memory_policy(intent_data: dict) -> bool:
    """Determine if we need to send full memory policy (write operations only)"""
    if intent_data.get("primary_intent") != "memory":
        return False
    
    memory_intent = intent_data.get("memory_intent", {})
    intent_type = memory_intent.get("intent", "")
    
    # Memory policy needed only for WRITE operations
    write_operations = {
        "add_core_identity", 
        "remove_core_identity", 
        "promotion",
        "add_symbol_belief"
    }
    
    return intent_type in write_operations

def get_intent_type(intent_data: dict) -> str:
    """Extract intent type from intent data"""
    if not intent_data:
        return "unknown"
    
    primary = intent_data.get("primary_intent")
    if primary == "memory":
        return intent_data.get("memory_intent", {}).get("intent", "unknown")
    elif primary == "financial":
        return intent_data.get("financial_intent", {}).get("intent", "unknown")
    
    return primary or "unknown"

def log_prompt_optimization(intent_data: dict, sent_constitution: bool, 
                           sent_memory_policy: bool, sent_financial: bool,
                           history_count: int):
    """Log what prompts we're sending for optimization monitoring"""
    intent_type = get_intent_type(intent_data)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "intent_type": intent_type,
        "sent_constitution": sent_constitution,
        "sent_memory_policy": sent_memory_policy,
        "sent_financial": sent_financial,
        "history_count": history_count
    }
    
    # Log to optimization tracking file
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "prompt_optimization_log.csv")
        
        # Write header if file doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("timestamp,intent_type,constitution,memory_policy,financial,history\n")
        
        with open(log_path, "a") as f:
            f.write(f"{log_entry['timestamp']},{intent_type},"
                    f"{int(sent_constitution)},{int(sent_memory_policy)},"
                    f"{int(sent_financial)},{history_count}\n")
    except Exception as e:
        print(f"Failed to log optimization: {e}")
    
    print(f"📊 Prompt optimization: intent={intent_type}, "
          f"constitution={sent_constitution}, "
          f"memory_policy={sent_memory_policy}, "
          f"financial={sent_financial}, "
          f"history={history_count}")

# ================================
# ROUTELLM CLIENT - LAZY LOADED
# ================================
_openai_client = None

def get_openai_client():
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=os.environ.get("ROUTELLM_API_KEY", "dummy-key"),   # Abacus key
            base_url="https://routellm.abacus.ai/v1"  # ⬅️ BASE ONLY
        )
        print("✅ Lazy loaded OpenAI client")
    return _openai_client

# ================================
# 👆 CACHE SYSTEM 👆
# ================================

class ArchitecturalCache:
    """
    Universal caching system for the entire application.
    Caches: LLM responses, market data, calculations, API calls, etc.
    """
    
    def __init__(self, default_ttl: int = 60, max_size: int = 1000):
        self._cache = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            self._stats['misses'] += 1
            return None
        
        entry = self._cache[key]
        
        # Check expiry
        if time.time() > entry['expires_at']:
            del self._cache[key]
            self._stats['misses'] += 1
            return None
        
        self._stats['hits'] += 1
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        # Evict oldest if at max size
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['created_at'])
            del self._cache[oldest_key]
            self._stats['evictions'] += 1
        
        self._cache[key] = {
            'value': value,
            'created_at': time.time(),
            'expires_at': time.time() + (ttl or self._default_ttl)
        }
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'size': len(self._cache),
            'evictions': self._stats['evictions']
        }


# Global cache instances
intent_cache = ArchitecturalCache(default_ttl=30, max_size=100)
market_data_cache = ArchitecturalCache(default_ttl=60, max_size=500)
indicator_cache = ArchitecturalCache(default_ttl=120, max_size=200)
llm_cache = ArchitecturalCache(default_ttl=300, max_size=50)


def clear_all_caches():
    """Clear all application caches."""
    intent_cache.invalidate()
    market_data_cache.invalidate()
    indicator_cache.invalidate()
    llm_cache.invalidate()
    print("🗑️ All caches cleared")


def get_cache_stats() -> dict:
    """Get statistics for all caches."""
    return {
        'intent_cache': intent_cache.get_stats(),
        'market_data_cache': market_data_cache.get_stats(),
        'indicator_cache': indicator_cache.get_stats(),
        'llm_cache': llm_cache.get_stats()
    }


def invalidate_symbol_cache(symbol: str):
    """Invalidate all cached data for a specific symbol."""
    market_data_cache.invalidate(symbol)
    indicator_cache.invalidate(symbol)
    print(f"🗑️ Cleared cache for {symbol}")

def normalize_text(text: str) -> str:
    """Normalize text for cache keys."""
    if not text:
        return ""
    # Lowercase, strip, remove extra whitespace
    return " ".join(text.lower().strip().split())

def stable_context_key(context: dict) -> str:
    """Create stable context key for caching."""
    if not context:
        return ""
    # Only use stable fields
    last_symbol = context.get("last_symbol", "")
    last_intent = context.get("last_intent", "")
    return f"{last_symbol}:{last_intent}"

def make_cache_key(prefix: str, *parts) -> str:
    """Create stable cache key from parts."""
    clean_parts = []
    for p in parts:
        if isinstance(p, str):
            clean_parts.append(normalize_text(p))
        elif isinstance(p, dict):
            # Sort dict items for consistent ordering
            clean_parts.append(str(sorted(p.items()) if p else ""))
        elif p is None:
            clean_parts.append("")
        else:
            clean_parts.append(str(p))
    return f"{prefix}:{'|'.join(clean_parts)}"

# ================================
# CORE LLM FUNCTIONS
# ================================
def optimized_routellm_think(user_input, working_memory, core_identity, 
                            conversation_history=None, intent_data=None,
                            first_call_of_session=False):
    """Optimized LLM reasoning with proper separation of concerns"""
    
    messages = []
    
    # Extract intent type for conditional logic
    intent_type = get_intent_type(intent_data)
    
    # 1. Determine what system prompts to include
    send_full_constitution = should_send_full_constitution(intent_data, first_call_of_session)
    send_memory_policy = should_send_memory_policy(intent_data)
    send_financial = (intent_data.get("primary_intent") == "financial") if intent_data else False
    
    if send_full_constitution:
        # Always send constitution for memory operations
        messages.append({"role": "system", "content": FULL_CONSTITUTION})
        
        # Only send memory policy for WRITE operations
        if send_memory_policy:
            messages.append({"role": "system", "content": FULL_MEMORY_POLICY})
        else:
            # For read-only queries, just mention we can access memory
            messages.append({"role": "system", "content": "You can access stored user facts when needed."})
    else:
        # Minimal system prompt for non-memory conversations
        messages.append({"role": "system", "content": SYSTEM_CORE})
    
    # 2. Only include financial protocol if explicitly financial intent
    if send_financial:
        messages.append({"role": "system", "content": FULL_FINANCIAL_INTELLIGENCE})
    # ⚠️ NO KEYWORD FALLBACK - rely only on intent classification
    
    # 3. Add memory context (simplified)
    if send_full_constitution:
        # For memory operations, provide facts count
        fact_count = len(core_identity.get("facts", []))
        if fact_count > 0:
            messages.append({
                "role": "system", 
                "content": f"You have access to {fact_count} stored personal facts."
            })
    
    # 4. Add conversation history (optimized for intent type)
    if conversation_history:
        # More context for memory operations, less for others
        if send_full_constitution:
            # Memory operations benefit from more context
            history_count = 4
        elif send_financial:
            # Financial queries need recent context for follow-ups
            history_count = 3
        else:
            # General chat - minimal context
            history_count = 2
        
        messages.extend(conversation_history[-history_count:])
    
    # 5. Add current user message
    messages.append({"role": "user", "content": user_input})
    
    # Debug logging for optimization monitoring
    log_prompt_optimization(intent_data, send_full_constitution, send_memory_policy, 
                           send_financial, len(conversation_history) if conversation_history else 0)
    
    # Call LLM
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="route-llm",
            messages=messages,
            temperature=0.6,
        )
        
        # MARK CONSTITUTION AS SENT AFTER FIRST SUCCESSFUL CALL
        if not st.session_state.constitution_sent and send_full_constitution:
            st.session_state.constitution_sent = True
            print(f"🏷️ Constitution marked as sent for session {st.session_state.session_id}")
        
        # Track LLM calls
        st.session_state.llm_calls_count = st.session_state.get('llm_calls_count', 0) + 1
        
        return resp.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return "I'm currently rate-limited. Please try again in a moment."
        return f"Error: {str(e)[:100]}"

def routellm_think_with_image(user_input, working_memory, core_identity, image_data=None):
    """Enhanced version that supports image input - OPTIMIZED."""
    
    # Use unified intent extraction (cached)
    context = st.session_state.get('financial_context', {})
    unified_intent = cached_extract_unified_intent(user_input, context)
    
    # Determine if this is first call of session
    first_call_of_session = not st.session_state.get('constitution_sent', False)
    
    # Build optimized messages
    messages = []
    
    # Determine what to send based on intent
    intent_type = get_intent_type(unified_intent)
    send_full_constitution = should_send_full_constitution(unified_intent, first_call_of_session)
    send_memory_policy = should_send_memory_policy(unified_intent)
    send_financial = (unified_intent.get("primary_intent") == "financial") if unified_intent else False
    
    if send_full_constitution:
        messages.append({"role": "system", "content": FULL_CONSTITUTION})
        if send_memory_policy:
            messages.append({"role": "system", "content": FULL_MEMORY_POLICY})
        else:
            messages.append({"role": "system", "content": "You can access stored user facts when needed."})
    else:
        messages.append({"role": "system", "content": SYSTEM_CORE})
    
    if send_financial:
        messages.append({"role": "system", "content": FULL_FINANCIAL_INTELLIGENCE})
    
    # Add memory context if needed
    if send_full_constitution:
        fact_count = len(core_identity.get("facts", []))
        if fact_count > 0:
            messages.append({
                "role": "system", 
                "content": f"You have access to {fact_count} stored personal facts."
            })
    
    # Add user message with image
    if image_data:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}"
                    }
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": user_input})
    
    # Log optimization
    log_prompt_optimization(unified_intent, send_full_constitution, send_memory_policy, 
                           send_financial, 0)  # No conversation history for images
    
    try:
        client = get_openai_client()
        
        # Use vision model if image present
        if image_data:
            # Check if we should use a vision model
            model = "gpt-4-vision-preview" if "vision" in os.environ.get("OPENAI_MODEL", "").lower() else "route-llm"
        else:
            model = "route-llm"
        
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
        )
        
        # Track session state
        if not st.session_state.constitution_sent and send_full_constitution:
            st.session_state.constitution_sent = True
            print(f"🏷️ Constitution marked as sent (with image) for session {st.session_state.get('session_id', 'unknown')}")
        
        # Track LLM calls
        st.session_state.llm_calls_count = st.session_state.get('llm_calls_count', 0) + 1
        
        return resp.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        print(f"LLM with Image Error: {error_msg}")
        
        if "rate limit" in error_msg.lower():
            return "I'm currently rate-limited. Please try again in a moment."
        elif "authentication" in error_msg.lower():
            return "Authentication error. Please check API configuration."
        elif "image" in error_msg.lower() or "vision" in error_msg.lower():
            return "I'm having trouble processing the image. Please try a different image or describe it in text."
        else:
            return f"I'm having trouble processing your request with the image. Please try again."

# ================================
# UTILITY FUNCTIONS
# ================================

def enhanced_get_market_data(symbol, timeframe="1d", source="auto"):
    """
    Enhanced market data fetcher - FIXED VERSION
    """
    print(f"🔍 Fetching market data for: {symbol}")

    # LAZY LOAD market pipelines
    lazy_load_market_pipeline()
    
    # LAZY LOAD: Check professional pipeline
    global PROFESSIONAL_PIPELINE_AVAILABLE, professional_pipeline
    if PROFESSIONAL_PIPELINE_AVAILABLE is None:
        try:
            from financial.data.professional_pipeline import ProfessionalMarketPipeline
            professional_pipeline = ProfessionalMarketPipeline(db_path="financial/data/professional.db")
            PROFESSIONAL_PIPELINE_AVAILABLE = True
            print(f"✅ Lazy loaded professional pipeline")
        except ImportError as e:
            PROFESSIONAL_PIPELINE_AVAILABLE = False
            print(f"⚠️ Professional pipeline import failed: {e}")
    
    # If professional pipeline is available, use it
    if PROFESSIONAL_PIPELINE_AVAILABLE and professional_pipeline:
        try:
            # FIX: Remove 'timeframe' parameter if pipeline doesn't accept it
            # Check what parameters the pipeline accepts
            data = professional_pipeline.fetch_market_data(
                symbol=symbol,
                interval=timeframe,  # Use 'interval' instead of 'timeframe'
                source=source
            )
            
            if data and data.get("latest_price"):
                # Ensure all fields are present
                result = {
                    'symbol': symbol,
                    'price': float(data.get("latest_price", 0)),
                    'open': float(data.get('open', data.get("latest_price", 0))),
                    'high': float(data.get('high', data.get("latest_price", 0))),
                    'low': float(data.get('low', data.get("latest_price", 0))),
                    'volume': int(data.get('volume', 0)),
                    'timestamp': data.get('timestamp', datetime.now().isoformat()),
                    'source': data.get('source', 'professional_pipeline'),
                    'status': 'ok',
                    'data': data.get('data', []),
                    'interval': data.get('interval', timeframe),
                    'realtime': data.get('realtime', False)
                }
                
                return result
                
        except Exception as e:
            print(f"Professional pipeline fetch failed: {e}")
            # Fall through to yfinance
    
    # Fallback to existing minimal pipeline
    if MARKET_PIPELINE_AVAILABLE and market_pipeline:
        try:
            data = market_pipeline.fetch_market_data(symbol, source=source)
            if data and data.get("latest_price"):
                return {
                    'symbol': symbol,
                    'price': float(data.get("latest_price", 0)),
                    'source': data.get("source", "pipeline"),
                    'status': 'ok',
                    'data': data.get("data", []),
                    'interval': data.get("interval", timeframe)
                }
        except Exception as e:
            print(f"Minimal pipeline fetch failed: {e}")
            # Fall through to yfinance
    
    # Ultimate fallback to yfinance (always works)
    try:
        # LAZY LOAD yfinance
        global _yfinance_module
        if _yfinance_module is None:
            import yfinance as yf
            _yfinance_module = yf
            print("✅ Lazy loaded yfinance")
        
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith(('.NS', '.BO', '^')) and source != "crypto":
            symbol_to_fetch = f"{symbol}.NS"
        else:
            symbol_to_fetch = symbol
        
        # Map timeframe
        period_map = {
            "1min": "1d", "5min": "5d", "15min": "5d", "30min": "5d",
            "1h": "5d", "1d": "5d", "1w": "1mo", "1M": "3mo"
        }
        
        interval_map = {
            "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
            "1h": "60m", "1d": "1d", "1w": "1wk", "1M": "1mo"
        }
        
        period = period_map.get(timeframe, "5d")
        interval = interval_map.get(timeframe, "1d")
        
        ticker = _yfinance_module.Ticker(symbol_to_fetch)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return {'symbol': symbol, 'price': 0, 'source': 'yfinance', 'status': 'no_data'}
        
        latest = df.iloc[-1]
        
        return {
            'symbol': symbol,
            'price': float(latest.get('Close', 0)),
            'open': float(latest.get('Open', 0)),
            'high': float(latest.get('High', 0)),
            'low': float(latest.get('Low', 0)),
            'volume': int(latest.get('Volume', 0)),
            'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name),
            'source': 'yfinance',
            'status': 'ok',
            'interval': interval,
            'realtime': False
        }
        
    except Exception as e:
        print(f"YFinance fetch failed: {e}")
        return {'symbol': symbol, 'price': 0, 'source': 'error', 'status': f'error: {str(e)[:100]}'}

def detect_market_query(text: str):
    """Detect if user is asking for market data"""
    text_lower = text.lower()
    
    market_keywords = [
        'price of', 'price for', 'what is', 'how much is',
        'stock price', 'share price', 'nifty', 'sensex',
        'reliance', 'tcs', 'infy', 'hdfc', 'icici',
        'market', 'trading', 'invest', 'buy', 'sell'
    ]
    
    # Simple symbol detection (you can enhance this)
    common_symbols = {
        'nifty': '^NSEI',
        'banknifty': '^NSEBANK', 
        'sensex': '^BSESN',
        'reliance': 'RELIANCE.NS',
        'tcs': 'TCS.NS',
        'infosys': 'INFY.NS',
        'hdfc bank': 'HDFCBANK.NS',
        'icici bank': 'ICICIBANK.NS'
    }
    
    detected = []
    
    # Check for specific symbols
    for name, symbol in common_symbols.items():
        if name in text_lower:
            detected.append({
                'name': name,
                'symbol': symbol,
                'type': 'stock' if '.NS' in symbol else 'index'
            })
    
    # Check for general market queries
    if any(keyword in text_lower for keyword in market_keywords):
        if not detected:  # General market query without specific symbol
            detected.append({
                'name': 'market',
                'symbol': '^NSEI',  # Default to Nifty
                'type': 'index'
            })
    
    return detected if detected else None

def handle_market_query_intelligent(user_input: str, current_mode: str) -> dict:
    """Intelligently handle financial queries - SMART OVERRIDE"""
    
    # LAZY LOAD financial tools
    lazy_load_financial_tools()
    
    if not MARKET_TOOLS_AVAILABLE or not financial_tools:
        return {
            "response": "Financial tools are not available.",
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    # Initialize context if not exists
    if 'financial_context' not in st.session_state:
        st.session_state.financial_context = {}
    
    # Extract financial intent
    fin_intent = extract_financial_intent(user_input)
    intent_type = fin_intent.get("intent")
    symbol = fin_intent.get("symbol")  # Extract symbol here
    
    # ALWAYS save symbol context for ALL financial queries
    if symbol:
        st.session_state.financial_context['last_symbol'] = symbol
        st.session_state.financial_context['last_intent'] = intent_type
        print(f"💾 Saved context: {symbol}")
    
    # Route to appropriate handler
    if intent_type == "basic_price":
        data = enhanced_get_market_data(symbol)
        
        if data.get('status') == 'ok':
            return {
                "response": f"📊 {symbol}: ₹{data['price']:.2f} (Source: {data['source']})",
                "status": UI_STATUS,
                "mode": current_mode
            }
        else:
            return {
                "response": f"❌ Could not fetch data for {symbol}",
                "status": UI_STATUS,
                "mode": current_mode
            }
    
    elif intent_type == "technical_analysis":
        # If no symbol from intent, check context
        if not symbol:
            symbol = st.session_state.financial_context.get('last_symbol')
        
        if not symbol:
            return {
                "response": "❌ Please specify a symbol. Example: 'RSI of INFY'",
                "status": UI_STATUS,
                "mode": current_mode
            }
        
        response_parts = [f"📊 TECHNICAL INDICATORS: {symbol}\n" + "="*50]
        
        # Get requested indicators
        indicators = fin_intent.get("indicators", [])
        if "rsi" in indicators:
            rsi = financial_tools.calculate_rsi(symbol)
            if rsi:
                signal = "Overbought 🔴" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral 🟡"
                response_parts.append(f"\nRSI(14): {rsi:.1f} - {signal}")
            else:
                response_parts.append(f"\n❌ Could not calculate RSI for {symbol}")
        
        if "ma" in indicators:
            mas = financial_tools.calculate_moving_averages(symbol)
            if mas:
                response_parts.append("\nMoving Averages:")
                for period, value in mas.items():
                    response_parts.append(f"  MA{period}: ₹{value:.2f}")
            else:
                response_parts.append("\n❌ Could not calculate Moving Averages")
        
        if "volatility" in indicators:
            vol = financial_tools.calculate_volatility(symbol)
            if vol:
                risk = "High 🔴" if vol > 40 else "Medium 🟡" if vol > 20 else "Low 🟢"
                response_parts.append(f"\nVolatility: {vol:.1f}% - {risk}")
            else:
                response_parts.append("\n❌ Could not calculate Volatility")
        
        # Add current price if no indicators specified
        if not indicators:
            data = enhanced_get_market_data(symbol)
            if data.get('status') == 'ok':
                response_parts.append(f"\nCurrent Price: ₹{data['price']:.2f}")
            response_parts.append("\nℹ️ Specify indicators like: RSI, Moving Averages, or Volatility")
        
        return {
            "response": "\n".join(response_parts),
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    elif intent_type == "comprehensive_report":
        if not symbol:
            symbol = st.session_state.financial_context.get('last_symbol')
            
        if not symbol:
            return {
                "response": "❌ Please specify a symbol. Example: 'Comprehensive report on INFY'",
                "status": UI_STATUS,
                "mode": current_mode
            }
            
        print(f"📊 Generating comprehensive report for {symbol}...")
        
        # Use generate_comprehensive_report method
        report = financial_tools.generate_comprehensive_report(symbol)
        
        return {
            "response": report,
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    elif intent_type == "comparison":
        symbols = fin_intent.get("symbols", [])
        
        # Handle "compare with TCS" when only one symbol is provided
        if len(symbols) == 1:
            last_symbol = st.session_state.financial_context.get('last_symbol')
            if last_symbol:
                symbols.insert(0, last_symbol)
        
        if len(symbols) < 2:
            return {
                "response": "❌ Please specify at least two symbols to compare. Example: 'Compare INFY and TCS'",
                "status": UI_STATUS,
                "mode": current_mode
            }
        
        response_parts = [f"📊 COMPARISON: {' vs '.join(symbols)}\n" + "="*50]
        
        for sym in symbols:
            data = enhanced_get_market_data(sym)
            if data.get('status') == 'ok':
                response_parts.append(f"\n{sym}:")
                response_parts.append(f"  Price: ₹{data['price']:.2f}")
                response_parts.append(f"  Source: {data.get('source', 'unknown')}")
                
                # Add basic technicals
                rsi = financial_tools.calculate_rsi(sym)
                if rsi:
                    response_parts.append(f"  RSI: {rsi:.1f}")
            else:
                response_parts.append(f"\n❌ {sym}: Data unavailable")
        
        return {
            "response": "\n".join(response_parts),
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    # If no handler matched, return None to fall through to main LLM
    return None
  
def check_goal_progress(user_input: str, vector_memory):
    """Check if user mentions goal progress"""
    if not vector_memory:
        return
    
    goal_keywords = ['goal', 'target', 'achieved', 'progress', 'completed', 'milestone', 'objective']
    
    if any(keyword in user_input.lower() for keyword in goal_keywords):
        # This could trigger goal-related follow-up
        pass

def update_goal_from_conversation(user_input: str, ai_response: str, vector_memory):
    """Update goals based on conversation"""
    if not vector_memory:
        return
    
    # Check for goal completion statements
    completion_phrases = [
        'i completed', 'i finished', 'i achieved', 'done with', 
        'accomplished', 'reached my goal'
    ]
    
    user_lower = user_input.lower()
    
    for phrase in completion_phrases:
        if phrase in user_lower:
            # Update goal progress in database
            try:
                conn = sqlite3.connect(vector_memory.db_path)  # <-- This needs sqlite3 imported
                cursor = conn.cursor()
                
                # Find relevant active goal (simplified - in reality would use NLP)
                cursor.execute('''
                    SELECT id, description FROM goals 
                    WHERE current_status = 'active' 
                    ORDER BY created_at DESC LIMIT 1
                ''')
                
                goal = cursor.fetchone()
                if goal:
                    # Mark as completed
                    cursor.execute('''
                        UPDATE goals 
                        SET progress = 1.0, current_status = 'completed', updated_at = ?
                        WHERE id = ?
                    ''', (datetime.now().isoformat(), goal[0]))
                    
                    conn.commit()
                    print(f"✓ Goal marked as completed: {goal[1]}")
                
                conn.close()
            except Exception as e:
                print(f"Error updating goal: {e}")
            break

def auto_journal_trading(user_input, model_response):
    """Auto-journal trading-related conversations."""
    if not model_response:
        return
    
    journal_path = os.path.join(MEMORY_DIR, "trading_journal.json")
    journal = load_json(journal_path, {"entries": []})
    
    text = (user_input + " " + str(model_response)).lower()
    
    emotions = []
    biases = []
    
    # Emotion detection
    if any(k in text for k in ["fear", "panic", "scared"]):
        emotions.append("fear")
    if any(k in text for k in ["greed", "excited", "overconfident"]):
        emotions.append("greed")
    
    # Bias detection
    if any(k in text for k in ["chased", "revenge"]):
        biases.append("revenge_trading")
    if any(k in text for k in ["rule", "discipline"]):
        biases.append("discipline_violation")
    
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_input": user_input,
        "summary": model_response[:300],
        "emotions": emotions,
        "biases": biases
    }
    
    journal["entries"].append(entry)
    save_json(journal_path, journal)

def mood_auto_detection(user_input: str, vector_memory):
    """Automatically detect and log mood from user input"""
    if not vector_memory:
        return
    
    user_lower = user_input.lower()
    
    # Mood keyword detection
    mood_keywords = {
        'happy': (8, 7, 'happy'),
        'excited': (9, 8, 'excited'),
        'good': (7, 6, 'happy'),
        'great': (8, 7, 'happy'),
        'awesome': (9, 8, 'excited'),
        'sad': (3, 4, 'sad'),
        'angry': (2, 3, 'angry'),
        'frustrated': (3, 4, 'angry'),
        'tired': (4, 3, 'tired'),
        'exhausted': (3, 2, 'tired'),
        'energetic': (8, 9, 'excited'),
        'calm': (7, 6, 'calm'),
        'stressed': (3, 4, 'anxious'),
        'anxious': (3, 4, 'anxious'),
        'nervous': (4, 5, 'anxious')
    }
    
    detected_moods = []
    for word, (mood_score, energy_level, emotion) in mood_keywords.items():
        if word in user_lower:
            detected_moods.append((mood_score, energy_level, emotion))
    
    if detected_moods:
        # Take the most extreme mood detected
        primary_mood = max(detected_moods, key=lambda x: abs(x[0] - 5))
        
        # Log mood
        vector_memory.log_mood(
            mood_score=primary_mood[0],
            energy_level=primary_mood[1],
            emotion=primary_mood[2],
            context="auto-detected from conversation",
            notes=f"From input: '{user_input[:50]}...'"
        )
        
        print(f"✓ Auto-detected mood: {primary_mood[2]} (score: {primary_mood[0]})")

def extract_intent(user_input: str) -> dict:
    """Extract intent from user input using LLM."""
    prompt = f"""
You are an intent and memory-structuring engine for a personal AI system.

Your job is to analyze the user's message and return a SINGLE JSON object
that strictly follows ONE of the schemas below.

You MUST understand meaning, not keywords.
You MUST NOT guess facts.
You MUST NOT answer the user.
You MUST return VALID JSON ONLY.

==============================
INTENT TYPES (CHOOSE ONE)
==============================

1. normal_chat
Use when the message does NOT request memory storage,
memory removal, preference changes, or factual recall.

Schema:
{{ "intent": "normal_chat" }}

------------------------------

2. add_core_identity
Use when the user states a stable personal identity
or relationship fact about themselves or their personal world.

This includes explicit memory requests AND
clear factual statements that describe stable relationships.

You MUST extract the information into structured form.

Schema:
{{
  "intent": "add_core_identity",
  "domain": "<personal | trading>",
  "entity": "<entity>",
  "attribute": "<attribute>",
  "value": "<value>"

}}

Examples:
User: "Remember my wife's name is Roohi"
→ {{
    "intent": "add_core_identity",
    "entity": "wife",
    "attribute": "name",
    "value": "Roohi"
  }}

User: "Save that I trade crypto"
→ {{
    "intent": "add_core_identity",
    "entity": "self",
    "attribute": "trades",
    "value": "crypto"
  }}
User: "Sam is my friend"
→ {{
    "intent": "add_core_identity",
    "entity": "friend",
    "attribute": "name",
    "value": "Sam"
  }}


------------------------------

3. remove_core_identity
Use ONLY when the user explicitly asks to forget,
remove, or correct a stored fact.

Schema:
{{
  "intent": "remove_core_identity",
  "entity": "<entity>",
  "attribute": "<attribute>"
}}

------------------------------

4. query_identity
Use when the user asks for a factual detail about themselves
or their stored personal information.

Schema:
{{
  "intent": "query_identity",
  "domain": "<personal | trading>",
  "entity": "<entity>",
  "attribute": "<attribute>"
}}
Examples:

User: "What is my wife's name?"
→ {{
  "intent": "query_identity",
  "domain": "personal",
  "entity": "wife",
  "attribute": "name"
}}

User: "What is my strategy?"
→ {{
  "intent": "query_identity",
  "domain": "trading",
  "entity": "strategy",
  "attribute": "risk_profile"
}}


------------------------------

5. set_preference
Use when the user asks for a persistent change
in how the assistant behaves or responds.

Schema:
{{
  "intent": "set_preference",
  "entity": "assistant",
  "attribute": "<preference_type>",
  "value": "<preference_value>"
}}

------------------------------

6. remove_preference
Use when the user asks to undo a previously set preference.

Schema:
{{
  "intent": "remove_preference",
  "entity": "assistant",
  "attribute": "<preference_type>"
}}

7. add_symbol_belief
Use when the user expresses a belief, bias, or conviction
about a specific market instrument or symbol.

This is NOT a diary entry.
This is NOT a personal identity fact.

Examples:
User: "I think NIFTY looks weak"
User: "I'm bullish on BTC"
User: "BANKNIFTY feels overextended"

Schema:
{{
  "intent": "add_symbol_belief",
  "symbol": "<instrument>",
  "belief": "<belief_statement>",
  "confidence": "<0.0 - 1.0 optional>"
}}


==============================
IMPORTANT RULES
==============================

-ATTRIBUTES ARE NOT LIMITED.

The "attribute" field is an open-ended semantic label inferred from meaning.
You are NOT restricted to attributes shown in examples.
Choose the most appropriate attribute name based on the user's intent.

Examples are illustrative, not exhaustive.


- Do NOT invent entities or attributes.
- If the message is ambiguous, choose normal_chat.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include extra keys.
- Output MUST be valid JSON.
- Exactly ONE intent per message.


==============================
USER MESSAGE
==============================
\"\"\"{user_input}\"\"\"
"""               
    
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="route-llm",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classification engine for personal facts.\n"
                        "Some facts are naturally SINGLE (wife, mother, father).\n"
                        "Some facts are naturally MULTIPLE (friends, hobbies, beliefs).\n"
                        "Your job is to decide plurality using common human understanding.\n"
                        "Do NOT assume facts are single by default."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Intent extraction error: {str(e)}")
        return {"intent": "normal_chat"}  # Safe fallback
    
def extract_unified_intent(user_input: str, context: dict = None) -> dict:
    """
    UNIFIED intent extraction for BOTH memory and financial queries.
    Replaces extract_intent() + extract_financial_intent() with ONE call.
    """
    
    # Quick context check for follow-ups
    user_lower = user_input.lower()
    if context and context.get('last_symbol'):
        last_symbol = context['last_symbol']
        
        # Fast path for obvious follow-ups
        if any(word in user_lower for word in ['rsi', 'moving average', 'ma', 'volatility']) and len(user_input.split()) < 5:
            indicators = []
            if 'rsi' in user_lower:
                indicators.append('rsi')
            if 'moving average' in user_lower or 'ma' in user_lower:
                indicators.append('ma')
            if 'volatility' in user_lower:
                indicators.append('volatility')
            
            return {
                "primary_intent": "financial",
                "financial_intent": {
                    "intent": "technical_analysis",
                    "symbol": last_symbol,
                    "indicators": indicators or ['rsi']
                }
            }
    
    prompt = f"""
You are a unified intent classifier for a personal AI system with financial capabilities.

Analyze this message and return STRICT JSON with ONE of these structures:

User Message: "{user_input}"
Context: Last discussed symbol = {context.get('last_symbol') if context else 'None'}

OUTPUT SCHEMA (choose ONE):

1. FINANCIAL QUERY:
{{
  "primary_intent": "financial",
  "financial_intent": {{
    "intent": "basic_price" | "technical_analysis" | "comprehensive_report" | "comparison" | "risk_analysis",
    "symbol": "<symbol or null>",
    "indicators": ["rsi", "ma", "volatility"] (only for technical_analysis),
    "symbols": ["SYM1", "SYM2"] (only for comparison)
  }}
}}

2. MEMORY OPERATION:
{{
  "primary_intent": "memory",
  "memory_intent": {{
    "intent": "add_core_identity" | "query_identity" | "remove_core_identity",
    "entity": "<entity>",
    "attribute": "<attribute>",
    "value": "<value>",
    "domain": "personal" | "trading"
  }}
}}

3. NORMAL CONVERSATION:
{{
  "primary_intent": "normal_chat"
}}

RULES:
- "analyze [STOCK]" → comprehensive_report
- "RSI of [STOCK]" → technical_analysis
- "price of [STOCK]" → basic_price
- "Remember X" → add_core_identity
- Follow-ups use context symbol

Return ONLY valid JSON.
"""
    
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="route-llm",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = json.loads(resp.choices[0].message.content)
        print(f"🎯 Unified intent: {result.get('primary_intent')}")
        return result
    except Exception as e:
        print(f"Intent extraction error: {e}")
        return {"primary_intent": "normal_chat"}

def extract_financial_intent(user_input: str) -> dict:
    """Extract financial analysis intent from user input - FIXED VERSION"""
    
    # Check if this is a follow-up query
    user_lower = user_input.lower()
    
    # Follow-up indicator detection
    follow_up_indicators = ['rsi', 'moving average', 'ma', 'volatility', 'technical']
    
    # If it's a follow-up query without a symbol
    if ('what about' in user_lower or 'show me' in user_lower or 
        'tell me' in user_lower) and any(indicator in user_lower for indicator in follow_up_indicators):
        
        # Get the last symbol from context
        last_symbol = st.session_state.get('financial_context', {}).get('last_symbol')
        
        if last_symbol:
            # Extract indicators
            indicators = []
            if 'rsi' in user_lower:
                indicators.append('rsi')
            if 'moving average' in user_lower or 'ma' in user_lower:
                indicators.append('ma')
            if 'volatility' in user_lower:
                indicators.append('volatility')
            
            return {
                "intent": "technical_analysis",
                "symbol": last_symbol,
                "indicators": indicators if indicators else ['rsi']  # Default to RSI
            }
    
    prompt = f"""
You are a financial query classifier for an AI trading assistant.

Analyze this user query and return STRICT JSON ONLY.

User Query: "{user_input}"

Classify into ONE of these intents:

1. basic_price - User wants just current price
   Schema: {{"intent": "basic_price", "symbol": "<symbol>"}}

2. technical_analysis - User wants ONLY specific technical indicators (RSI, MA, etc.)
   Schema: {{"intent": "technical_analysis", "symbol": "<symbol>", "indicators": ["rsi", "ma", "volatility"]}}
   **ONLY use this if user asks for SPECIFIC indicators like "RSI of INFY"**

3. comprehensive_report - User wants detailed/full analysis including volume profile, momentum, risk metrics
   Keywords: comprehensive, full, detailed, complete, technical analysis, report
   Schema: {{"intent": "comprehensive_report", "symbol": "<symbol>"}}

4. comparison - User wants to compare multiple stocks
   Schema: {{"intent": "comparison", "symbols": ["SYMBOL1", "SYMBOL2"]}}

5. risk_analysis - User wants risk metrics only
   Schema: {{"intent": "risk_analysis", "symbol": "<symbol>"}}

6. market_overview - General market question
   Schema: {{"intent": "market_overview"}}

7. not_financial - Not a financial query
   Schema: {{"intent": "not_financial"}}

**CRITICAL RULES:**
- "technical analysis" → comprehensive_report (not technical_analysis intent)
- "comprehensive report" → comprehensive_report
- "full analysis" → comprehensive_report
- "detailed analysis" → comprehensive_report
- "RSI of INFY" → technical_analysis (only RSI)
- "moving averages of INFY" → technical_analysis (only MA)
- "What about RSI?" → technical_analysis (use context symbol)
- "Compare with TCS" → comparison (use context + TCS)

Examples:
- "What's RELIANCE price?" → {{"intent": "basic_price", "symbol": "RELIANCE"}}
- "Give me RSI of TCS" → {{"intent": "technical_analysis", "symbol": "TCS", "indicators": ["rsi"]}}
- "Comprehensive report on INFY" → {{"intent": "comprehensive_report", "symbol": "INFY"}}
- "Technical analysis of INFY" → {{"intent": "comprehensive_report", "symbol": "INFY"}}
- "Full analysis of INFY" → {{"intent": "comprehensive_report", "symbol": "INFY"}}
- "Compare RELIANCE and TCS" → {{"intent": "comparison", "symbols": ["RELIANCE", "TCS"]}}
- "What about RSI?" (after INFY discussion) → {{"intent": "technical_analysis", "symbol": "INFY", "indicators": ["rsi"]}}
- "Compare with TCS" (after INFY discussion) → {{"intent": "comparison", "symbols": ["INFY", "TCS"]}}

Return ONLY valid JSON. No explanation.
"""
    
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="route-llm",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = json.loads(resp.choices[0].message.content)
        print(f"🔍 Financial intent detected: {result}")  # Debug log
        return result
    except Exception as e:
        print(f"Financial intent extraction error: {e}")
        return {"intent": "not_financial"}
    
# ================================
# 👆 END OF OPTIMIZED HANDLER 👆
# ================================

# Update cached_extract_unified_intent:
def cached_extract_unified_intent(user_input: str, context: dict = None) -> dict:
    """Cached version of intent extraction."""
    # SKIP CACHE for very short social utterances BUT STILL CACHE THE RESULT
    if len(user_input.split()) <= 3:
        words = user_input.lower().split()
        small_talk = {"hi", "hello", "hey", "ok", "good", "yes", "no", "thanks", "thank"}
        if any(word in small_talk for word in words):
            print("⏩ Skipping LLM for small talk")
            # Create cache key for small talk
            cache_key = make_cache_key("intent_small", user_input)
            cached = intent_cache.get(cache_key)
            if cached:
                print("⚡ Small talk cache HIT")
                return cached
            
            # Return predefined intent for small talk
            result = {"primary_intent": "normal_chat"}
            intent_cache.set(cache_key, result, ttl=300)
            return result
    
    # Use stable cache key for regular queries
    cache_key = make_cache_key("intent", user_input, context)
    
    cached = intent_cache.get(cache_key)
    if cached:
        print("⚡ Intent cache HIT")
        return cached
    
    print("🔄 Intent cache MISS - calling LLM")
    result = extract_unified_intent(user_input, context)
    intent_cache.set(cache_key, result, ttl=300)
    return result
# Update cached_market_data:
def cached_market_data(symbol: str, timeframe: str = "1d") -> dict:
    """Cached version of market data fetch."""
    # Normalize symbol
    normalized_symbol = symbol.upper().strip()
    cache_key = make_cache_key("market", normalized_symbol, timeframe)
    
    cached = market_data_cache.get(cache_key)
    if cached:
        print(f"⚡ Market data cache HIT for {normalized_symbol}")
        return cached
    
    print(f"🔄 Market data cache MISS for {normalized_symbol}")
    result = enhanced_get_market_data(symbol, timeframe)
    market_data_cache.set(cache_key, result, ttl=60)
    return result

# Update cached_financial_tool_call:
def cached_financial_tool_call(method_name: str, symbol: str, *args, **kwargs) -> Any:
    """Cached wrapper for financial tool calls (RSI, MA, etc.)"""
    # Normalize symbol
    normalized_symbol = symbol.upper().strip()
    cache_key = make_cache_key("indicator", method_name, normalized_symbol, args, kwargs)
    
    cached = indicator_cache.get(cache_key)
    if cached:
        print(f"⚡ Indicator cache HIT: {method_name}({normalized_symbol})")
        return cached
    
    print(f"🔄 Indicator cache MISS: {method_name}({normalized_symbol})")
    
    lazy_load_financial_tools()
    
    if not financial_tools:
        return None
    
    method = getattr(financial_tools, method_name)
    result = method(symbol, *args, **kwargs)
    
    indicator_cache.set(cache_key, result, ttl=120)
    return result    

def get_latest_identity_audit(entity: str, attribute: str):
    """Get latest identity audit entry."""
    path = os.path.join(MEMORY_DIR, "promotion_audit.json")
    data = load_json(path, {"events": []})
    
    for e in reversed(data.get("events", [])):
        if e.get("entity") == entity and e.get("attribute") == attribute:
            return e
    
    return None

def explain_non_promotion(audit_event: dict) -> str:
    """Explain why an identity wasn't promoted."""
    if not audit_event:
        return "I don't have enough consistent information yet."
    
    reason = audit_event.get("reason")
    
    explanations = {
        "threshold_not_met": "I've only seen this mentioned once so far, so I'm waiting for more confirmation.",
        "conflict_window": "I've seen conflicting information recently, so I'm waiting for things to settle.",
        "inactive_candidate": "This was mentioned earlier, but it hasn't come up again for a while.",
        "confidence_decay": "This information hasn't been reinforced recently, so I'm less confident about it."
    }
    
    return explanations.get(reason, "I'm still observing before recording this.")

def log_promotion_audit(entry: dict):
    """Log promotion audit entry."""
    try:
        path = os.path.join(MEMORY_DIR, "promotion_audit.json")
        audit = load_json(path, {"events": []})
        
        entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
        audit["events"].append(entry)
        
        with open(path, "w") as f:
            json.dump(audit, f, indent=2)
    except Exception as e:
        print(f"Failed to log promotion audit: {e}")

def promote_identity_candidates(working_memory):
    """Promote identity candidates to core identity."""
    updated = False
    remaining = []
    
    now = datetime.now()
    
    for o in working_memory.get("observations", []):
        if o.get("type") != "identity_candidate":
            remaining.append(o)
            continue
        
        if o.get("active") is False:
            log_promotion_audit({
                "result": "blocked",
                "entity": o["entity"],
                "attribute": o["attribute"],
                "value": o["value"],
                "confidence": o.get("confidence"),
                "count": o.get("count"),
                "reason": "inactive_candidate"
            })
            remaining.append(o)
            continue
        
        # --- Conflict suppression ---
        conflict = False
        o_last = datetime.fromisoformat(o["last_seen"])
        
        for other in working_memory.get("observations", []):
            if (other is not o and 
                other.get("type") == "identity_candidate" and
                other["entity"] == o["entity"] and
                other["attribute"] == o["attribute"] and
                other["value"] != o["value"]):
                
                other_last = datetime.fromisoformat(other["last_seen"])
                if abs(o_last - other_last) <= IDENTITY_CONFLICT_WINDOW:
                    conflict = True
                    break
        
        if conflict:
            log_promotion_audit({
                "result": "blocked",
                "entity": o["entity"],
                "attribute": o["attribute"],
                "value": o["value"],
                "confidence": o.get("confidence"),
                "count": o.get("count"),
                "reason": "conflict_window"
            })
            remaining.append(o)
            continue
        
        # --- Promotion eligibility ---
        if (o.get("confidence", 0) >= 0.75 or o.get("count", 0) >= 2):
            action = {
                "type": "ADD_FACT",
                "domain": "personal",
                "entity": o["entity"],
                "attribute": o["attribute"],
                "value": o["value"],
                "owner": "self",
                "source": "promotion",
                "confidence": o["confidence"]
            }
            
            apply_memory_action(action)
            updated = True
            
            log_promotion_audit({
                "result": "promoted",
                "entity": o["entity"],
                "attribute": o["attribute"],
                "value": o["value"],
                "confidence": o.get("confidence"),
                "count": o.get("count"),
                "reason": "thresholds_met"
            })
        else:
            log_promotion_audit({
                "result": "blocked",
                "entity": o["entity"],
                "attribute": o["attribute"],
                "value": o["value"],
                "confidence": o.get("confidence"),
                "count": o.get("count"),
                "reason": "threshold_not_met"
            })
            remaining.append(o)
    
    if updated:
        working_memory["observations"] = remaining
        save_json(os.path.join(MEMORY_DIR, "working_memory.json"), working_memory)
    
    return updated

def detect_mode(user_input):
    """Detect current mode from user input."""
    text = user_input.lower()
    
    trading_keywords = [
        "trade", "trading", "market", "bias",
        "journal", "pnl", "loss", "profit",
        "setup", "entry", "exit"
    ]
    
    if any(word in text for word in trading_keywords):
        return "trading"
    
    return "personal"

def is_greeting_only(text: str) -> bool:
    """Check if text is just a greeting."""
    t = text.lower().strip()
    return t in {"hi", "hello", "hey", "hiya"}

def clean_markdown(text: str) -> str:
    """Clean markdown from text."""
    if not text:
        return text
    
    replacements = [
        ("**", ""),      # remove bold
        ("##", ""),      # remove headings
        ("###", ""),
        ("####", ""),
        ("#", ""),
        ("**", ""),
    ]
    
    for a, b in replacements:
        text = text.replace(a, b)
        text = text.replace("•", "-")
    
    return text.strip()

def clean_ai_output(text: str) -> str:
    """Clean AI output text."""
    if not text:
        return text
    
    bad_tokens = ["<s>", "</s>", "<S>", "[OUT]"]
    for t in bad_tokens:
        text = text.replace(t, "")
        text = clean_markdown(text)
    
    return text.strip()

def is_clarification_only(text: str) -> bool:
    """Check if text is just asking for clarification."""
    t = text.lower().strip()
    
    clarification_phrases = [
        "what do you mean",
        "can you explain",
        "i don't understand",
        "please clarify",
        "clarify that",
        "explain that",
        "could you clarify"
    ]
    
    return any(p in t for p in clarification_phrases)

def is_casual_chat(text: str) -> bool:
    """Check if text is casual chat."""
    t = text.lower().strip()
    
    casual_triggers = [
        "joke",
        "funny",
        "make my day",
        "cheer me up",
        "how are you",
        "hello",
        "hi",
        "hey",
        "who are you",
        "what can you do",
        "are you dumb",
        "are you smart",
        "tell me something",
        "say something"
    ]
    
    return any(trigger in t for trigger in casual_triggers)

import re

def decide_cardinality_llm(entity: str, attribute: str, value: str, existing_facts: list, user_input: str) -> dict:
    """LLM decides cardinality of a new fact."""
    prompt = f"""
You are deciding how a new personal fact should be handled.

Existing facts (same entity + attribute):
{json.dumps(existing_facts, indent=2)}

New statement:
"{user_input}"

Proposed new fact:
entity = "{entity}"
attribute = "{attribute}"
value = "{value}"

Decide ONE of the following actions:
1. conflict  → single-cardinality, incompatible
2. append    → valid additional fact
3. duplicate → same fact already exists

Output STRICT JSON ONLY in this schema:

{{
  "cardinality": "single" | "multiple",
  "action": "conflict" | "append" | "duplicate"
}}

Do NOT explain.
Do NOT add extra keys.
"""
    
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="route-llm",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classification engine for personal facts.\n"
                        "Some relationships are naturally SINGLE (wife, mother, father).\n"
                        "Some relationships are naturally MULTIPLE (friends, hobbies, beliefs).\n"
                        "Use common human understanding.\n"
                        "Do NOT assume identity facts are single by default."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        
        decision = json.loads(resp.choices[0].message.content)
        return decision
    except Exception as e:
        print(f"Cardinality decision error: {str(e)}")
        # ultra-safe fallback: assume single & conflict
        return {
            "cardinality": "single",
            "action": "conflict"
        }

def requires_live_price(text: str) -> bool:
    """Check if text requires live price data."""
    t = text.lower()
    
    triggers = [
        r"\bprice\b",
        r"\bltp\b",
        r"\bquote\b",
        r"\bvalue\b",
        r"\bright now\b",
        r"\blive\b",
        r"\bcurrent\b",
        r"\bwhere is\b",
    ]
    
    return any(re.search(p, t) for p in triggers)

def handle_identity_confirmation():
    """Handle identity confirmation."""
    global PENDING_IDENTITY_CONFIRMATION
    info = PENDING_IDENTITY_CONFIRMATION
    
    if not info:
        return {
            "response": "There is nothing pending to confirm.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    candidates = info.get("candidates", [])
    
    if not candidates:
        return {
            "response": "There is nothing pending to confirm.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    best = sorted(
        candidates,
        key=lambda c: (c.get("confidence", 0), c.get("last_seen", "")),
        reverse=True
    )[0]
    
    apply_memory_action({
        "type": "ADD_FACT",
        "domain": "personal",
        "entity": info["entity"],
        "attribute": info["attribute"],
        "value": best["value"],
        "owner": "self",
        "source": "user_confirmation",
        "confidence": 1.0
    })
    
    # --- clear pending identity state ---
    working_memory["observations"] = [
        o for o in working_memory.get("observations", [])
        if not (
            o.get("type") == "identity_candidate"
            and o.get("entity") == info["entity"]
            and o.get("attribute") == info["attribute"]
        )
    ]
    
    save_json(os.path.join(MEMORY_DIR, "working_memory.json"), working_memory)
    PENDING_IDENTITY_CONFIRMATION = None
    
    return {
        "response": "Confirmed.",
        "status": UI_STATUS,
        "mode": CURRENT_MODE
    }

def handle_quick_confirmation():
    """Handle quick confirmations without LLM call"""
    global PENDING_IDENTITY_CONFIRMATION
    
    info = PENDING_IDENTITY_CONFIRMATION
    if not info or not info.get("candidates"):
        return {
            "response": "Nothing to confirm.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    # Get best candidate
    candidate = sorted(
        info["candidates"],
        key=lambda c: (c.get("confidence", 0), c.get("last_seen", "")),
        reverse=True
    )[0]
    
    # Apply without LLM
    apply_memory_action({
        "type": "ADD_FACT",
        "domain": "personal",
        "entity": info["entity"],
        "attribute": info["attribute"],
        "value": candidate["value"],
        "owner": "self",
        "source": "user_confirmation",
        "confidence": 1.0
    })
    
    # Clear pending
    PENDING_IDENTITY_CONFIRMATION = None
    
    return {
        "response": f"Confirmed: {candidate['value']}",
        "status": UI_STATUS,
        "mode": CURRENT_MODE
    }

def handle_financial_query_optimized(fin_intent: dict, current_mode: str) -> dict:
    """
    OPTIMIZED financial query handler with caching.
    Direct routing - no LLM needed!
    """
    lazy_load_financial_tools()
    
    if not MARKET_TOOLS_AVAILABLE or not financial_tools:
        return {
            "response": "Financial tools unavailable.",
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    intent_type = fin_intent.get('intent')
    symbol = fin_intent.get('symbol')
    
    # Check context if no symbol
    if not symbol and 'financial_context' in st.session_state:
        symbol = st.session_state.financial_context.get('last_symbol')
    
    if not symbol and intent_type != 'market_overview':
        return {
            "response": "Please specify a symbol.",
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    # Route to handlers with caching
    if intent_type == "basic_price":
        data = cached_market_data(symbol)
        if data.get('status') == 'ok':
            return {
                "response": f"📊 {symbol}: ₹{data['price']:.2f} ({data['source']})",
                "status": UI_STATUS,
                "mode": current_mode,
                "last_symbol": symbol  # Add this for context
            }
        else:
            return {
                "response": f"❌ Could not fetch data for {symbol}",
                "status": UI_STATUS,
                "mode": current_mode,
                "last_symbol": symbol  # Add this for context
            }
    
    elif intent_type == "technical_analysis":
        indicators = fin_intent.get('indicators', [])
        response_parts = [f"📊 {symbol} TECHNICAL INDICATORS\n{'='*50}"]
        
        if "rsi" in indicators:
            rsi = cached_financial_tool_call('calculate_rsi', symbol)
            if rsi:
                signal = "Overbought 🔴" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral 🟡"
                response_parts.append(f"\nRSI(14): {rsi:.1f} - {signal}")
            else:
                response_parts.append(f"\n❌ Could not calculate RSI for {symbol}")
        
        if "ma" in indicators:
            mas = cached_financial_tool_call('calculate_moving_averages', symbol)
            if mas:
                response_parts.append("\nMoving Averages:")
                for period, value in mas.items():
                    response_parts.append(f"  MA{period}: ₹{value:.2f}")
            else:
                response_parts.append("\n❌ Could not calculate Moving Averages")
        
        if "volatility" in indicators:
            vol = cached_financial_tool_call('calculate_volatility', symbol)
            if vol:
                risk = "High 🔴" if vol > 40 else "Medium 🟡" if vol > 20 else "Low 🟢"
                response_parts.append(f"\nVolatility: {vol:.1f}% - {risk}")
            else:
                response_parts.append("\n❌ Could not calculate Volatility")
        
        # Add current price if no indicators specified
        if not indicators:
            data = cached_market_data(symbol)
            if data.get('status') == 'ok':
                response_parts.append(f"\nPrice: ₹{data['price']:.2f}")
            response_parts.append("\nℹ️ Specify indicators like: RSI, Moving Averages, or Volatility")
        
        return {
            "response": "\n".join(response_parts),
            "status": UI_STATUS,
            "mode": current_mode,
            "last_symbol": symbol  # Add this for context
        }
    
    elif intent_type == "comprehensive_report":
        # Use proper cache key
        cache_key = make_cache_key("comprehensive", symbol, "full_report")
        cached = indicator_cache.get(cache_key)
        
        if cached:
            print(f"⚡ Comprehensive report cache HIT: {symbol}")
            return {
                "response": cached,
                "status": UI_STATUS,
                "mode": current_mode,
                "last_symbol": symbol  # Important for context!
            }
        
        print(f"🔄 Generating comprehensive report: {symbol}")
        report = financial_tools.generate_comprehensive_report(symbol)
        
        # Cache with appropriate TTL (2 minutes)
        indicator_cache.set(cache_key, report, ttl=120)
        
        # Update LLM cache stats
        llm_cache_key = make_cache_key("llm_comprehensive", symbol)
        llm_cache.set(llm_cache_key, True, ttl=120)
        print(f"💾 Cached comprehensive report for {symbol}")
        
        return {
            "response": report,
            "status": UI_STATUS,
            "mode": current_mode,
            "last_symbol": symbol  # Important for context!
        }
    
    elif intent_type == "comparison":
        symbols = fin_intent.get('symbols', [])
        response_parts = [f"📊 COMPARISON: {' vs '.join(symbols)}\n{'='*50}"]
        
        for sym in symbols:
            data = cached_market_data(sym)
            if data.get('status') == 'ok':
                response_parts.append(f"\n{sym}:")
                response_parts.append(f"  Price: ₹{data['price']:.2f}")
                
                rsi = cached_financial_tool_call('calculate_rsi', sym)
                if rsi:
                    response_parts.append(f"  RSI: {rsi:.1f}")
            else:
                response_parts.append(f"\n❌ {sym}: Data unavailable")
        
        return {
            "response": "\n".join(response_parts),
            "status": UI_STATUS,
            "mode": current_mode,
            "last_symbol": symbols[-1] if symbols else None  # Use last symbol for context
        }
    
    elif intent_type == "risk_analysis":
        # Fallback to comprehensive report for now
        return {
            "response": f"Risk analysis for {symbol} would go here.\n(Use comprehensive report for detailed analysis)",
            "status": UI_STATUS,
            "mode": current_mode,
            "last_symbol": symbol  # Add this for context
        }
    
    elif intent_type == "market_overview":
        return {
            "response": "Market overview would go here.",
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    elif intent_type == "not_financial":
        return {
            "response": "This doesn't appear to be a financial query.",
            "status": UI_STATUS,
            "mode": current_mode
        }
    
    return None

# ================================
# MAIN PROCESSING FUNCTION
# ================================
def process_user_input(user_input: str, conversation_history: list = None) -> dict:
    """Main function to process user input."""
    global PENDING_IDENTITY_CONFIRMATION, CURRENT_MODE, UI_STATUS
    
    if conversation_history is None:
        conversation_history = []
    
    # Ensure session is initialized
    ensure_session_initialized() 
    
    ai_response = None
    model_response = None
    image_data = None
    
    # --- FILE HANDLING ---
    if hasattr(st.session_state, 'uploaded_file') and st.session_state.uploaded_file:
        file_data = st.session_state.uploaded_file
        processed = process_file(file_data)
        
        # Format the input with file content
        user_input, image_data = format_file_for_llm(processed, user_input)
        
        # Clear the file from session state
        st.session_state.uploaded_file = None
    
    lower = user_input.lower().strip()
    
    # --- Quick confirmation handling ---
    if PENDING_IDENTITY_CONFIRMATION and lower in {
        "yes", "confirm", "correct", "that's right", "right", "yes confirm", "yes that's correct"
    }:
        return handle_quick_confirmation()

    # --- Implicit confirmation handling ---
    if PENDING_IDENTITY_CONFIRMATION:
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model="route-llm",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You detect whether a user message confirms previously "
                            "mentioned information. Answer ONLY yes or no."
                        ),
                    },
                    {"role": "user", "content": user_input},
                ],
                temperature=0,
            )
            
            decision = resp.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Confirmation detection error: {str(e)}")
            decision = "no"
        
        if decision.startswith("yes"):
            return handle_identity_confirmation()
    
    # --- confirmation-based identity promotion ---
    if lower in {
        "yes", "yes that's correct", "yes that's correct",
        "correct", "that's right", "confirm", "confirm that"
    }:
        if PENDING_IDENTITY_CONFIRMATION:
            info = PENDING_IDENTITY_CONFIRMATION
            candidates = info.get("candidates")
            
            if not candidates:
                return {
                    "response": "There's nothing pending that needs confirmation.",
                    "status": UI_STATUS,
                    "mode": CURRENT_MODE
                }
            
            # pick strongest candidate
            candidate = sorted(
                candidates,
                key=lambda o: (o.get("confidence", 0), o.get("last_seen", "")),
                reverse=True
            )[0]
            
            action = {
                "type": "ADD_FACT",
                "domain": "personal",
                "entity": info["entity"],
                "attribute": info["attribute"],
                "value": candidate["value"],
                "owner": "self",
                "source": "user_confirmation",
                "confidence": 1.0
            }
            
            apply_memory_action(action)
            
            # remove all pending candidates for this slot
            working_memory["observations"] = [
                o for o in working_memory.get("observations", [])
                if not (
                    o.get("type") == "identity_candidate"
                    and o.get("entity") == info["entity"]
                    and o.get("attribute") == info["attribute"]
                )
            ]
            
            save_json(os.path.join(MEMORY_DIR, "working_memory.json"), working_memory)
            PENDING_IDENTITY_CONFIRMATION = None
            
            return {
                "response": f"Got it. I've updated that to {candidate['value']}.",
                "status": UI_STATUS,
                "mode": CURRENT_MODE
            }
    
    # --- Slash commands ---
    if lower.startswith("/"):
        return {"response": "Command not supported", "status": UI_STATUS, "mode": CURRENT_MODE}
    
    # For compatibility with auto_learn, extract basic intent
    # but use cached version when possible
    if "remember" in lower or "save" in lower or "from now on" in lower:
        # Use old system for explicit memory commands to maintain compatibility
        intent = extract_intent(user_input)
    else:
        # Use unified intent for better caching and optimization
        intent = cached_extract_unified_intent(user_input, {})
        
        # Convert unified intent to old format for auto_learn
        if intent.get("primary_intent") == "memory":
            memory_intent = intent.get("memory_intent", {})
            intent = {
                "intent": memory_intent.get("intent", "normal_chat"),
                "entity": memory_intent.get("entity", ""),
                "attribute": memory_intent.get("attribute", ""),
                "value": memory_intent.get("value", ""),
                "domain": memory_intent.get("domain", "personal")
            }
        else:
            intent = {"intent": "normal_chat"}
    
    # Default domain for memory ops
    if intent.get("intent") == "add_core_identity":
        intent.setdefault("domain", CURRENT_MODE)
    
    # Implicit name rule for people
    if intent.get("intent") == "add_core_identity":
        if not intent.get("attribute") and intent.get("entity") in PERSON_ENTITIES:
            intent["attribute"] = "name"
    
    # --- Plurality guard ---
    if (intent.get("intent") == "add_core_identity" and
        intent.get("entity") == "friend" and
        "also" in lower):
        auto_learn(user_input, working_memory)
        return {
            "response": "Okay.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    # --- WRITE ---
    if intent["intent"] == "add_core_identity" and (
        "remember" in lower or "save" in lower or "from now on" in lower):
        
        domain = "personal"
        action = {
            "type": "ADD_FACT",
            "domain": domain,
            "entity": intent["entity"],
            "attribute": intent["attribute"],
            "value": intent["value"],
            "owner": "self"
        }
        
        memory = load_identity_memory()
        
        existing_facts = [
            f for f in memory.get("facts", [])
            if (f["entity"] == intent["entity"] and f["attribute"] == intent["attribute"])
        ]
        
        decision = decide_cardinality_llm(
            entity=intent["entity"],
            attribute=intent["attribute"],
            value=intent["value"],
            existing_facts=existing_facts,
            user_input=user_input
        )
        
        # CARDINALITY ENFORCEMENT
        if decision["action"] == "conflict":
            return {
                "response": (
                    f"I already have this recorded as "
                    f"'{existing_facts[-1]['value']}'. "
                    f"If this is incorrect, please say "
                    f"'Update my {intent['entity']}'s {intent['attribute']}'."
                ),
                "status": UI_STATUS,
                "mode": CURRENT_MODE
            }
        
        if decision["action"] == "duplicate":
            return {
                "response": "Okay.",
                "status": UI_STATUS,
                "mode": CURRENT_MODE
            }
        
        action["cardinality"] = decision["cardinality"]
        result = apply_memory_action(action)
        
        if result.get("conflict"):
            return {
                "response": (
                    f"I already have this recorded as "
                    f"'{result['existing_value']}'. "
                    f"If this is incorrect, please say "
                    f"'Update my {intent['entity']}'s {intent['attribute']}'."
                ),
                "status": UI_STATUS,
                "mode": CURRENT_MODE
            }
        
        if result.get("applied"):
            return {
                "response": "Okay, I've saved that.",
                "status": UI_STATUS,
                "mode": CURRENT_MODE
            }
        
        return {
            "response": "I decided not to save that.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    elif intent["intent"] == "add_core_identity":
        # Implicit identity → working memory only
        auto_learn(user_input, working_memory)
        return {
            "response": "Okay.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    elif intent["intent"] == "add_symbol_belief":
        action = {
            "type": "ADD_FACT",
            "domain": "trading",
            "entity": intent["symbol"],
            "attribute": "belief",
            "value": intent["belief"],
            "confidence": intent.get("confidence", 0.5),
            "owner": "self",
            "source": "user"
        }
        
        apply_memory_action(action)
        
        return {
            "response": "Belief noted.",
            "status": UI_STATUS,
            "mode": CURRENT_MODE
        }
    
    # --- READ (LOCKED) ---
    if intent["intent"] == "query_identity":
        PENDING_IDENTITY_CONFIRMATION = None
        
        if intent["domain"] == "personal":
            memory = load_identity_memory()
        else:
            memory = load_json(os.path.join(MEMORY_DIR, "state_memory.json"), {"states": []})
        
        value = query_fact(
            intent["domain"],
            intent["entity"],
            intent["attribute"],
            memory
        )
        
        # --- Collect all pending identity candidates ---
        pending_candidates = [
            o for o in working_memory.get("observations", [])
            if (o.get("type") == "identity_candidate" and
                o.get("entity") == intent["entity"] and
                o.get("attribute") == intent["attribute"] and
                o.get("active", True))
        ]
        
        if value is not None:
            response_text = value
            
            if pending_candidates:
                names = ", ".join(c["value"] for c in pending_candidates)
                response_text += (
                    f". However, you recently mentioned {names}, "
                    f"and I'm waiting for confirmation before updating."
                )
                
                PENDING_IDENTITY_CONFIRMATION = {
                    "entity": intent["entity"],
                    "attribute": intent["attribute"],
                    "candidates": pending_candidates
                }
        else:
            audit = get_latest_identity_audit(
                intent["entity"],
                intent["attribute"]
            )
            response_text = explain_non_promotion(audit)
            
            if audit and audit.get("value"):
                PENDING_IDENTITY_CONFIRMATION = {
                    "entity": intent["entity"],
                    "attribute": intent["attribute"],
                    "value": audit.get("value")
                }
            else:
                PENDING_IDENTITY_CONFIRMATION = None
            
            if pending_candidates:
                PENDING_IDENTITY_CONFIRMATION = {
                    "entity": intent["entity"],
                    "attribute": intent["attribute"],
                    "candidates": pending_candidates
                }
            else:
                PENDING_IDENTITY_CONFIRMATION = None
        
        return {
            "response": response_text,
            "status": UI_STATUS,
            "mode": CURRENT_MODE,
            "meta": {
                "pending": bool(pending_candidates)
            }
        }
    
        # --- Mode detection ---
    new_mode = detect_mode(user_input)
    if new_mode != CURRENT_MODE:
        CURRENT_MODE = new_mode
    
    # --- OPTIMIZED: UNIFIED INTENT + DIRECT ROUTING ---
    context = st.session_state.get('financial_context', {})
    unified_intent = cached_extract_unified_intent(user_input, context)

    primary_intent = unified_intent.get('primary_intent')

    # --- CONTEXT AWARENESS: Clear financial context if topic changed ---
    if primary_intent != "financial":
        # Clear financial intent but keep last symbol for reference
        if 'financial_context' in st.session_state:
            # Only clear the intent, keep the symbol for potential follow-ups
            st.session_state.financial_context['last_intent'] = None
            print("🧹 Cleared financial intent (topic changed)")
    else:
        # This is a financial query, save context
        fin_intent = unified_intent.get('financial_intent', {})
        symbol = fin_intent.get('symbol')
        if symbol:
            if 'financial_context' not in st.session_state:
                st.session_state.financial_context = {}
            st.session_state.financial_context['last_symbol'] = symbol
            st.session_state.financial_context['last_intent'] = fin_intent.get('intent')
            print(f"💾 Financial context saved: {symbol}")

    # Direct financial routing (skips main LLM!)
    if primary_intent == "financial":
        fin_intent = unified_intent.get('financial_intent', {})
        
        # Route to optimized handler
        result = handle_financial_query_optimized(fin_intent, CURRENT_MODE)
        if result:
            # Update financial context with last_symbol if provided
            if result.get('last_symbol'):
                if 'financial_context' not in st.session_state:
                    st.session_state.financial_context = {}
                st.session_state.financial_context['last_symbol'] = result['last_symbol']
                if 'last_intent' not in st.session_state.financial_context:
                    st.session_state.financial_context['last_intent'] = fin_intent.get('intent')
            return result

    # Direct financial routing (skips main LLM!)
    if primary_intent == "financial":
        fin_intent = unified_intent.get('financial_intent', {})
        
        # Save context
        symbol = fin_intent.get('symbol')
        if symbol:
            if 'financial_context' not in st.session_state:
                st.session_state.financial_context = {}
            st.session_state.financial_context['last_symbol'] = symbol
            st.session_state.financial_context['last_intent'] = fin_intent.get('intent')
            print(f"💾 Context saved: {symbol}")
        
        # Route to optimized handler
        result = handle_financial_query_optimized(fin_intent, CURRENT_MODE)
        if result:
            return result
    
    # --- Use session-based first call flag ---
    first_call_of_session = not st.session_state.constitution_sent
    
    # --- Main reasoning (LLM) ---
    try:
        response = optimized_routellm_think(
            user_input,
            working_memory,
            core_identity,
            conversation_history,
            unified_intent,  # Pass intent data
            first_call_of_session  # Pass session flag
        )
        ai_response = clean_ai_output(response)
        UI_STATUS = "Online" if ai_response else "Error"
    except Exception as e:
        UI_STATUS = "Error"
        import traceback
        traceback.print_exc()
        ai_response = f"LLM crashed – {str(e)[:100]}..."
    
    # --- ENHANCED MEMORY INTEGRATION ---
    # LAZY LOAD enhanced memory systems
    global ENHANCED_MEMORY_AVAILABLE, vector_memory, personality_engine, pattern_recognizer
    if ENHANCED_MEMORY_AVAILABLE is None:
        try:
            from memory.vector_memory import VectorMemory
            from memory.personality_engine import PersonalityEngine
            from memory.pattern_recognizer import PatternRecognizer
            
            # Initialize enhanced memory systems
            vector_memory = VectorMemory()
            personality_engine = PersonalityEngine()
            pattern_recognizer = PatternRecognizer()
            
            ENHANCED_MEMORY_AVAILABLE = True
            print("✅ Lazy loaded enhanced memory systems")
        except ImportError as e:
            ENHANCED_MEMORY_AVAILABLE = False
            print(f"⚠️ Enhanced memory not available: {e}")
    
    if ENHANCED_MEMORY_AVAILABLE:
        try:
            # 1. Store conversation in vector memory
            memory_id = vector_memory.add_conversation_memory(
                user_input=user_input,
                ai_response=ai_response,
                metadata={
                    "mode": CURRENT_MODE,
                    "status": UI_STATUS,
                    "intent": intent.get("intent", "unknown")
                }
            )
        
            # 2. Analyze for personality insights
            personality_engine.analyze_conversation(user_input, ai_response)
        
            # 3. Detect patterns
            pattern_recognizer.analyze_interaction(
                user_input=user_input,
                timestamp=datetime.now(),
                metadata={
                    "mode": CURRENT_MODE,
                    "intent": intent.get("intent", "unknown")
                }
            )
        
            # 4. Auto-detect mood from conversation
            mood_auto_detection(user_input, vector_memory)
        
            # 5. Search for similar past conversations (for context)
            if len(user_input) > 10:  # Only for substantial inputs
                similar_memories = vector_memory.search_similar_memories(user_input, n_results=2)
                if similar_memories and similar_memories['documents']:
                    # Could add context from past conversations to future responses
                    pass
                
        except Exception as e:
            print(f"Enhanced memory error: {e}")
            # Non-critical, continue with normal flow
    
    # --- Final safety guard: never return empty AI output ---
    if not ai_response or not ai_response.strip():
        ai_response = "I'm here. What would you like to work on?"
    
    # --- Final promotion pass ---
    promote_identity_candidates(working_memory)
    promote_to_preferences(working_memory, core_identity)
    
    return {
        "response": ai_response,
        "status": UI_STATUS,
        "mode": CURRENT_MODE
    }

def auto_learn(user_input, working_memory):
    """Auto-learn from user input - ENHANCED VERSION"""
    now = datetime.now().isoformat(timespec="seconds")
    
    # Try unified intent first, fall back to old intent
    try:
        # Use cached version if possible
        context = st.session_state.get('financial_context', {})
        unified_intent = cached_extract_unified_intent(user_input, context)
        
        if unified_intent.get("primary_intent") == "memory":
            memory_intent = unified_intent.get("memory_intent", {})
            intent = {
                "intent": memory_intent.get("intent", "normal_chat"),
                "entity": memory_intent.get("entity", ""),
                "attribute": memory_intent.get("attribute", ""),
                "value": memory_intent.get("value", ""),
                "domain": memory_intent.get("domain", "personal")
            }
        else:
            # Fall back to old intent system for compatibility
            intent = extract_intent(user_input)
    except Exception as e:
        print(f"Intent extraction error in auto_learn: {e}")
        intent = extract_intent(user_input)  # Fallback to old system

    # -----------------------------
    # ORIGINAL CONFIDENCE DECAY LOGIC
    # -----------------------------
    observations = working_memory.get("observations", [])
    updated = []
    
    now_dt = datetime.fromisoformat(now)
    
    for o in observations:
        last_seen = o.get("last_seen")
        if not last_seen:
            updated.append(o)
            continue
        
        last_dt = datetime.fromisoformat(last_seen)
        days_idle = (now_dt - last_dt).days
        
        if days_idle > 0:
            decay = days_idle * CONFIDENCE_DECAY_PER_DAY
            o["confidence"] = max(o.get("confidence", 0.6) - decay, 0.0)
        
        inactive_days = (now_dt - last_dt).days
        
        if o.get("confidence", 0) < CONFIDENCE_MIN_THRESHOLD:
            o["active"] = False
            o["inactive_since"] = now
        elif inactive_days >= SOFT_DELETE_AFTER_DAYS:
            o["active"] = False
            o["inactive_since"] = now
        else:
            o["active"] = True
            o.pop("inactive_since", None)
        
        updated.append(o)
    
    working_memory["observations"] = updated
    
    # -----------------------------
    # ENHANCED: PATTERN DETECTION
    # -----------------------------
    # LAZY LOAD enhanced memory systems
    global ENHANCED_MEMORY_AVAILABLE, pattern_recognizer
    if ENHANCED_MEMORY_AVAILABLE is None:
        try:
            from memory.pattern_recognizer import PatternRecognizer
            pattern_recognizer = PatternRecognizer()
            # Set flag but don't need to load all modules yet
            ENHANCED_MEMORY_AVAILABLE = True
        except ImportError:
            ENHANCED_MEMORY_AVAILABLE = False
    
    if ENHANCED_MEMORY_AVAILABLE and pattern_recognizer:
        try:
            # Extract intent for pattern analysis
            intent = extract_intent(user_input)
            
            pattern_recognizer.analyze_interaction(
                user_input=user_input,
                timestamp=datetime.fromisoformat(now),
                metadata={
                    "intent": intent.get("intent", "unknown"),
                    "source": "auto_learn"
                }
            )
        except Exception as e:
            print(f"Pattern analysis error in auto_learn: {e}")
    
    # -----------------------------
    # ORIGINAL IDENTITY CANDIDATE LOGIC
    # -----------------------------
    intent = extract_intent(user_input)
    
    if intent.get("intent") in ("add_symbol_belief", "add_diary_entry"):
        save_json(os.path.join(MEMORY_DIR, "working_memory.json"), working_memory)
        return
    
    if intent.get("intent") == "add_core_identity":
        obs = {
            "type": "identity_candidate",
            "entity": intent["entity"],
            "attribute": intent["attribute"],
            "value": intent["value"].strip().lower(),
            "count": 1,
            "first_seen": now,
            "last_seen": now,
            "confidence": 0.6
        }
        
        found = False
        for o in updated:
            if (o.get("type") == "identity_candidate" and
                o["entity"] == obs["entity"] and
                o["attribute"] == obs["attribute"] and
                o["value"] == obs["value"]):
                
                o["count"] += 1
                o["confidence"] = min(o["confidence"] + 0.15, 1.0)
                o["last_seen"] = now
                found = True
                break
        
        if not found:
            updated.append(obs)
        
        working_memory["observations"] = updated
    
    # Save ONCE at the end
    save_json(os.path.join(MEMORY_DIR, "working_memory.json"), working_memory)
    
    # -----------------------------
    # ENHANCED: PERSONALITY ANALYSIS
    # -----------------------------
    # LAZY LOAD personality engine
    global personality_engine
    if ENHANCED_MEMORY_AVAILABLE and personality_engine is None:
        try:
            from memory.personality_engine import PersonalityEngine
            personality_engine = PersonalityEngine()
        except ImportError:
            personality_engine = None
    
    if ENHANCED_MEMORY_AVAILABLE and personality_engine:
        try:
            # Simple personality trait extraction
            traits_to_check = {
                "detail_oriented": ["detail", "specific", "exactly", "precise"],
                "emotional": ["feel", "emotion", "happy", "sad", "angry"],
                "analytical": ["analyze", "data", "numbers", "statistics"],
                "practical": ["simple", "easy", "straightforward", "basic"]
            }
            
            user_lower = user_input.lower()
            for trait, keywords in traits_to_check.items():
                if any(keyword in user_lower for keyword in keywords):
                    personality_engine.update_trait(
                        trait_name=trait,
                        trait_value=0.7,
                        confidence=0.6
                    )
        except Exception as e:
            print(f"Personality analysis error: {e}")

def main():
    """Main Streamlit app function."""
    global CURRENT_MODE, UI_STATUS

    # Initialize session FIRST
    ensure_session_initialized()
    
    st.set_page_config(page_title="AI Agent", layout="wide")
    
    st.title("🤖 AI Agent")
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'show_upstox_auth' not in st.session_state:
        st.session_state.show_upstox_auth = False
    
    # Sidebar with Upstox authentication
    with st.sidebar:
        st.header("🔐 Upstox Authentication")
        
        # LAZY LOAD Upstox
        lazy_load_financial_tools()
        if UPSTOX_AVAILABLE is None:
            try:
                from financial.auth.upstox_auth import upstox_auth_flow, check_upstox_auth, UpstoxAuth
                UPSTOX_AVAILABLE = True
                print("✅ Lazy loaded Upstox auth")
            except ImportError as e:
                UPSTOX_AVAILABLE = False
                upstox_auth_flow = None
                check_upstox_auth = lambda: False
                UpstoxAuth = None
                print(f"⚠️ Upstox auth not available: {e}")
        
        if UPSTOX_AVAILABLE:
            if check_upstox_auth():
                st.success("✅ Upstox: Authenticated")
                if st.button("Logout from Upstox"):
                    from financial.auth.upstox_auth import UpstoxAuth
                    auth = UpstoxAuth()
                    auth.logout()
                    st.rerun()
            else:
                if st.button("Authenticate Upstox", type="primary"):
                    # Show auth flow in main area
                    st.session_state.show_upstox_auth = True
                    st.rerun()
        else:
            st.warning("Upstox not available")
            if st.button("Install Upstox"):
                st.code("pip install upstox-python")
        
        st.divider()
        st.header("Memory")
        
        # Simple mode selector
        mode = st.radio("Mode:", ["personal", "trading"])
        CURRENT_MODE = mode

        if st.button("Clear Cache"):
            clear_all_caches()
            st.success("Cache cleared!")

            with st.expander("📊 Cache Stats"):
                stats = get_cache_stats()
                for cache_name, cache_stats in stats.items():
                    st.text(f"{cache_name}: {cache_stats['hit_rate']} ({cache_stats['hits']} hits)")
            
            st.divider()
            st.header("Optimization Dashboard")
            
            with st.expander("🔍 Prompt Optimization Insights"):
                try:
                    log_path = os.path.join("logs", "prompt_optimization_log.csv")
                    if os.path.exists(log_path):
                        import pandas as pd
                        df = pd.read_csv(log_path)
                        
                        if not df.empty:
                            # Calculate stats
                            total_calls = len(df)
                            constitution_rate = (df['constitution'].sum() / total_calls * 100)
                            memory_policy_rate = (df['memory_policy'].sum() / total_calls * 100)
                            financial_rate = (df['financial'].sum() / total_calls * 100)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Calls", total_calls)
                            col2.metric("Constitution", f"{constitution_rate:.1f}%")
                            col3.metric("Memory Policy", f"{memory_policy_rate:.1f}%")
                            
                            # Session info
                            st.metric("Current Session", st.session_state.get('session_id', 'N/A')[:8])
                            st.metric("LLM Calls", st.session_state.get('llm_calls_count', 0))
                            st.metric("Constitution Sent", "Yes" if st.session_state.get('constitution_sent', False) else "No")
                            
                            # Intent distribution
                            st.subheader("Intent Distribution")
                            intent_counts = df['intent_type'].value_counts()
                            for intent, count in intent_counts.items():
                                st.text(f"{intent}: {count}")
                        else:
                            st.info("No optimization data yet.")
                    else:
                        st.info("Optimization logging not started yet.")
                except Exception as e:
                    st.error(f"Could not load optimization data: {e}")
            
            if st.button("Clear Conversation"):
                st.session_state.conversation = []
                st.rerun()
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'jpg', 'png'])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"Uploaded: {uploaded_file.name}")
    
    # Main area - show Upstox auth if requested
    if UPSTOX_AVAILABLE and st.session_state.get('show_upstox_auth', False):
        st.header("Upstox Authentication")
        
        # Try to show auth flow
        try:
            if upstox_auth_flow:
                upstox_auth_flow()
            else:
                st.error("Upstox auth flow not available")
        except Exception as e:
            st.error(f"Error in auth flow: {e}")
        
        if st.button("← Back to Chat"):
            st.session_state.show_upstox_auth = False
            st.rerun()
        
        return  # Don't show chat when authenticating
    
    # Normal chat interface
    st.caption(f"Mode: {CURRENT_MODE} | Status: {UI_STATUS} | Session: {st.session_state.get('session_id', '')[:8]}")
    
    # Main chat interface
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # User input
    if prompt := st.chat_input("What would you like to work on?"):
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": prompt})
        
        # Process input
        with st.spinner("Thinking..."):
            try:
                result = process_user_input(prompt, st.session_state.conversation)
                ai_response = result["response"]
            except Exception as e:
                ai_response = f"Error: {str(e)}"
            
            # Add AI response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": ai_response})
            
            # Update status
            if "status" in result:
                UI_STATUS = result["status"]
            if "mode" in result:
                CURRENT_MODE = result["mode"]
        
        st.rerun()


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    # Quick test to verify functions exist
    try:
        test_fyers_integration()
        
        # Test that required functions exist
        test_functions = [
            'get_market_data',
            'mood_auto_detection', 
            'enhanced_get_market_data',
            'process_user_input',
            'main'
        ]
        
        for func in test_functions:
            if func in globals():
                print(f"✅ {func}() exists")
            else:
                print(f"❌ {func}() MISSING!")
                
    except Exception as e:
        print(f"Setup error: {e}")
    
    print("\n🚀 Starting AI Agent...")
    main()