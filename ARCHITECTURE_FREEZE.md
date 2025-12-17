## Live Data vs LLM Semantics (FROZEN)

The following rules are frozen and MUST NOT be violated without an explicit architecture review.

• LLM connectivity is independent of market data availability  
• Market data availability is independent of LLM connectivity  
• UI MUST display both statuses explicitly  
• LLM Online ≠ Market Data Online  

Violations of these rules cause user trust failure and are considered architecture regressions.

🟡 Explain-Only Mode Active

Live market data is currently unavailable.
The AI is online and can explain strategies,
indicators, and trading concepts without
using real-time prices.



• Broker is primary
• Global data is fallback
• Health check controls switching
• UI shows active source
• Brain remains source-agnostic

• Broker = primary
• Twelve Data = fallback
• DataRouter is single gate
• UI shows active source
• Brain never chooses data source

• broker_fetch() exists in ONE place
• Router is the only switch
• Health marked only on broker success
• Providers never fallback themselves

broker_fetch(symbol, interval):
• returns broker data
• marks health ONLY on success
• raises exception on failure

• broker_fetch is single broker entry
• router imports broker_fetch
• providers never call each other

• broker_fetch exists in one place
• router controls switching
• health marked only on success
• providers are isolated
