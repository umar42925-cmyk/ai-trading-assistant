import streamlit as st
from main import process_user_input


st.set_page_config(
    page_title="AI Trading Assistant",
    layout="wide"
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

# --------------------------------------------------
# Market Data Source (single owner)
# --------------------------------------------------
if "market_source" not in st.session_state:
    st.session_state.market_source = "broker"  # default

source = st.session_state.market_source


# --------------------------------------------------
# Page config (must be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="AI Trading Assistant",
    layout="wide"
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "status" not in st.session_state:
    st.session_state.status = "Online"

if "mode" not in st.session_state:
    st.session_state.mode = "personal"

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("**Brain:** RouteLLM")
st.sidebar.markdown(f"**Mode:** {st.session_state.mode}")
st.sidebar.markdown(f"**Status:** {st.session_state.status}")
st.sidebar.markdown(f"📡 Market Data Source: {source.title()}")
st.sidebar.markdown("**Memory:** ON")

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.chat = []
    st.session_state.status = "Online"
    st.session_state.mode = "personal"
    st.rerun()

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("📈 AI Trading Assistant")

# --------------------------------------------------
# Input form
# --------------------------------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type a message")
    submitted = st.form_submit_button("Send")

# --------------------------------------------------
# Handle submission (THIS is the only place logic runs)
# --------------------------------------------------
if submitted and user_input:
    # Call core logic
    result = process_user_input(user_input)

    ai_text = result.get("response")
    status = result.get("status", "Unknown")
    mode = result.get("mode", st.session_state.mode)

    # Update sidebar state
    st.session_state.status = status
    st.session_state.mode = mode

    # Save chat
    st.session_state.chat.append(("You", user_input))

    if ai_text:
        st.session_state.chat.append(("AI", ai_text))
    else:
        st.session_state.chat.append(
            ("AI", f"⚠️ System status: {status}. Please try again shortly.")
        )

    # Correct rerun (NOT experimental)
    st.rerun()

# --------------------------------------------------
# Render chat
# --------------------------------------------------
st.markdown("---")

for speaker, text in st.session_state.chat:
    if speaker == "You":
        st.markdown("**👤 You**")
        st.markdown(f"> {text}")
    else:
        st.markdown("**🤖 AI**")
        st.markdown(text)
if source == "twelve_data":
    st.warning(
        "Broker data unavailable. Using global market data (Twelve Data) as fallback."
    )
