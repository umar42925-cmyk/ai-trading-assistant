import streamlit as st
from main import process_user_input

# --------------------------------------------------
# Page config (MUST be first Streamlit call)
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
# Sidebar (CLEAN, STATIC, SAFE)
# --------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("**Brain:** RouteLLM")
st.sidebar.markdown(f"**Mode:** {st.session_state.mode}")
st.sidebar.markdown(f"**Status:** {st.session_state.status}")
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
# Handle submission (ONLY logic entry point)
# --------------------------------------------------
if submitted and user_input:
    try:
        result = process_user_input(user_input)

        if not isinstance(result, dict):
            raise ValueError("Invalid response format from process_user_input")

        ai_text = result.get("response")
        status = result.get("status", "Online")
        mode = result.get("mode", st.session_state.mode)

    except Exception as e:
        ai_text = f"⚠️ Internal error: {e}"
        status = "Offline"
        mode = st.session_state.mode

    st.session_state.status = status
    st.session_state.mode = mode

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("AI", ai_text))

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
