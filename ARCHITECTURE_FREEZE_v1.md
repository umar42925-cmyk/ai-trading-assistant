🔒 AI Agent Architecture Freeze — v1.0
Status

FROZEN (v1.0)
Any change beyond this point must be intentional, scoped, and versioned.

1. What is frozen (DO NOT CHANGE casually)
1.1 Single-Brain Rule

Exactly one LLM reasoning path per user message

RouteLLM is the current brain

Model swaps must not change behavior

❌ No parallel LLM calls
❌ No “backup LLM thinking”

1.2 Memory Layering (Hard Boundary)

Policy Layer (DECIDES):

extract_intent()

cardinality decisions

conflict detection

confirmation logic

promotion rules

decay logic

audit logging

Primitive Layer (WRITES ONLY):

apply_memory_action()

apply_memory_action() MUST remain a dumb, low-level write primitive.
If policy appears there, it is a bug.

1.3 Memory Tiers (Schema Locked)

Working Memory → working_memory.json

Core Identity → core_identity.json

State Memory → state_memory.json

Specialized memory:

bias.json

trading_journal.json

promotion_audit.json

❌ No silent schema changes
❌ No cross-tier writes

1.4 Market Data Truth Chain

Market data must follow this order:

Professional pipeline (if available)

Minimal pipeline

yfinance (ultimate fallback)

Rules:

Never hallucinate prices

Never say “no access to live data”

If data fails → say unavailable

1.5 Financial Intelligence Separation

LLM → understands what the user wants

Tools → fetch prices, indicators, reports

LLM never pretends to fetch data

1.6 Constitution Supremacy

AGENT_CONSTITUTION

FINANCIAL_INTELLIGENCE

MEMORY_POLICY

These are system contracts, not suggestions.

They override:

UI behavior

model quirks

developer convenience

1.7 Stability Rules

No refactors “because it’s cleaner”

No feature bundles

One change → test → commit → freeze again