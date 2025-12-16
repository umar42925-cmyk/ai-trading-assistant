import streamlit as st
from main import process_user_input

if "status" not in st.session_state:
    st.session_state.status = None



st.sidebar.title("⚙️ Control Panel")

st.sidebar.markdown("**Brain:** OpenRouter")

mode = st.session_state.get("mode", "Auto-detect")
status = st.session_state.get("status", "Unknown")

st.sidebar.markdown(f"**Mode:** {mode}")
st.sidebar.markdown(f"**Status:** {status}")
st.sidebar.markdown("**Memory:** ON")

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.chat = []
    st.session_state.status = None
    st.session_state.mode = None
    st.experimental_rerun()



st.set_page_config(
    page_title="AI Trading Assistant",
    layout="wide"
)

st.title("📈 AI Trading Assistant")

from main import get_openrouter_api_key

if get_openrouter_api_key():
    st.sidebar.success("OpenRouter: Connected")
else:
    st.sidebar.error("OpenRouter: Missing API key")


# --- Chat state ---
if "chat" not in st.session_state:
    st.session_state.chat = []

# --- Input box ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type a message")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    result = process_user_input(prompt)

ai_text = result.get("response")
status = result.get("status", "Unknown")

if ai_text:
    st.session_state.messages.append(
        {"role": "assistant", "content": ai_text}
    )
else:
    # 🔥 THIS is what you are missing
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"⚠️ System status: {status}. Please try again shortly."
        }
    )



    st.session_state.status = result.get("status")
    st.session_state.mode = result.get("mode")


    st.session_state.chat.append(("You", user_input))

    if result.get("response"):
        st.session_state.chat.append(("AI", result["response"]))
    else:
        st.warning(f"Status: {result.get('status', 'Unknown')}")


# --- Render chat ---
st.markdown("---")

for speaker, text in st.session_state.chat:
    if speaker == "You":
        st.markdown(f"**👤 You**")
        st.markdown(f"> {text}")
    else:
        st.markdown(f"**🤖 AI**")
        st.markdown(text)
