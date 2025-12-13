import json
import os
from ollama import chat

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

memory = load_memory()

SYSTEM_PROMPT = f"""
You are my personal AI trading assistant.

Known long-term information about me:
{memory}

CORE IDENTITY:
{memory.get("identity")}

AVAILABLE FRAMEWORKS:
{memory.get("frameworks")}

CURRENT BIAS:
{memory.get("bias")}

BEHAVIOR RULES:
- Never give buy/sell signals
- Never predict price
- Do not assume Elliott Wave unless explicitly stated
- If Elliott Wave is used, enforce its rules
- If another framework is used, ask clarifying questions
- Always ask for invalidation when bias is stated
- Present alternate perspectives when analysis is subjective

RESPONSE STYLE RULES:
- Ask at most ONE clarifying question at a time
- Prioritize invalidation over confirmation
- Avoid long lists unless explicitly requested
- Be concise by default

REASONING STYLE:
- Think step-by-step internally, but respond concisely
- Prefer structured reasoning over verbosity
- If unsure, ask ONE precise question
- Avoid unnecessary elaboration
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("\n=== Local Trading Assistant (Phase 1) Started ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye 👋")
        break

    messages.append({"role": "user", "content": user_input})

    MAX_HISTORY = 6
    messages = messages[-MAX_HISTORY:]

    response = chat(
    model="qwen2.5:7b",
    messages=messages,
    options={
        "num_predict": 180
    }
)

    reply = response["message"]["content"]
    print("\nAI:", reply, "\n")

    messages.append({"role": "assistant", "content": reply})
