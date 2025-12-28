import streamlit as st
from main import process_user_input, working_memory
import json
from datetime import datetime

# --------------------------------------------------
# Page config (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="AI Personal Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS for better UI
# --------------------------------------------------
st.markdown("""
<style>
    /* Main chat container */
    .main {
        background-color: #0e1117;
        padding-bottom: 150px !important;
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat message styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        animation: slideInRight 0.3s ease;
    }
    
    .ai-message {
        background: #1e2127;
        color: #e8e8e8;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px 0;
    }
    
    .status-online {
        background: #10b981;
        color: white;
    }
    
    .status-offline {
        background: #ef4444;
        color: white;
    }
    
    .status-rate-limited {
        background: #f59e0b;
        color: white;
    }
    
    /* Mode badge */
    .mode-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 5px 0;
    }
    
    /* Fixed input container */
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 250px;
        right: 0;
        background-color: #0e1117;
        padding: 20px;
        border-top: 2px solid #667eea;
        z-index: 999;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        background-color: #1e2127 !important;
        color: #e8e8e8 !important;
        border: 2px solid #667eea !important;
        border-radius: 20px !important;
        padding: 12px 20px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 10px 30px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* File uploader - make it compact */
    [data-testid="stFileUploader"] {
        width: 60px !important;
    }
    
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] section > button {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        background: #667eea !important;
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] section > button:hover {
        background: #764ba2 !important;
    }
    
    /* Hide file uploader text */
    [data-testid="stFileUploader"] section > div {
        display: none !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1d24;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Thinking indicator */
    .thinking-indicator {
        position: fixed;
        bottom: 140px;
        left: 50%;
        transform: translateX(-50%);
        background: #1e2127;
        color: #e8e8e8;
        padding: 10px 20px;
        border-radius: 20px;
        border: 2px solid #667eea;
        z-index: 998;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "status" not in st.session_state:
    st.session_state.status = "Online"

if "mode" not in st.session_state:
    st.session_state.mode = "personal"

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

if "thinking" not in st.session_state:
    st.session_state.thinking = False

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # Status indicator
    status_color = {
        "Online": "status-online",
        "Offline": "status-offline",
        "Rate-limited": "status-rate-limited"
    }.get(st.session_state.status, "status-online")
    
    st.markdown(f"""
        <div class="status-badge {status_color}">
            ● {st.session_state.status}
        </div>
    """, unsafe_allow_html=True)
    
    # Mode indicator
    st.markdown(f"""
        <div class="mode-badge">
            📍 {st.session_state.mode.capitalize()} Mode
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System info
    st.markdown("**🧠 Brain:** RouteLLM")
    st.markdown("**💾 Memory:** ON")
    st.markdown(f"**💬 Messages:** {len(st.session_state.chat) // 2}")
    
    st.markdown("---")
    
    # Working memory status
    st.markdown("### 🔄 Working Memory")
    try:
        obs_count = len(working_memory.get("observations", []))
        st.markdown(f"**Observations:** {obs_count}")
        
        if obs_count > 0:
            st.caption("Learning from your patterns...")
    except Exception:
        pass
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()
    
    # Export chat button
    if st.button("💾 Export Chat", use_container_width=True):
        if st.session_state.chat:
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "messages": [
                    {"speaker": speaker, "text": text}
                    for speaker, text in st.session_state.chat
                ]
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(chat_export, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("🤖 AI Personal Agent")
st.caption("Your intelligent companion for personal insights and trading analysis")

# --------------------------------------------------
# Chat display area
# --------------------------------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.chat:
        # Welcome message
        st.markdown("""
            <div class="ai-message">
                👋 Hi! I'm your personal AI agent. I can help you with:
                <br><br>
                • Personal memory and preferences<br>
                • Trading analysis and market insights<br>
                • Task planning and decision support<br>
                <br>
                What would you like to explore today?
            </div>
        """, unsafe_allow_html=True)
    else:
        # Render chat messages
        for speaker, text in st.session_state.chat:
            if speaker == "You":
                st.markdown(f"""
                    <div class="user-message">
                        <strong>You</strong><br>
                        {text}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="ai-message">
                        <strong>🤖 AI Agent</strong><br>
                        {text}
                    </div>
                """, unsafe_allow_html=True)

# --------------------------------------------------
# Thinking indicator
# --------------------------------------------------
if st.session_state.thinking:
    st.markdown("""
        <div class="thinking-indicator">
            🤔 Thinking...
        </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Fixed input area at bottom
# --------------------------------------------------
input_container = st.container()

with input_container:
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([7, 0.5, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Type your message here... (Press Enter to send)",
                label_visibility="collapsed",
                key=f"user_input_{st.session_state.input_key}"
            )
        
        with col2:
            uploaded_file = st.file_uploader("📎", label_visibility="collapsed", key=f"file_{st.session_state.input_key}")
        
        with col3:
            send_button = st.form_submit_button("Send", use_container_width=True)

# --------------------------------------------------
# Handle submission
# --------------------------------------------------
if send_button:
    if user_input:
        # Add user message immediately
        st.session_state.chat.append(("You", user_input))
        st.session_state.thinking = True
        
        # Store uploaded file in session state
        if uploaded_file is not None:
            st.session_state.last_uploaded_file = {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "data": uploaded_file.getvalue()
            }
        else:
            st.session_state.last_uploaded_file = None
        
        st.rerun()
    elif uploaded_file is not None:
        # File only, no text
        st.session_state.chat.append(("You", f"[Uploaded: {uploaded_file.name}]"))
        st.session_state.thinking = True
        st.session_state.last_uploaded_file = {
            "name": uploaded_file.name,
            "type": uploaded_file.type,
            "data": uploaded_file.getvalue()
        }
        st.rerun()

# Process if thinking
if st.session_state.thinking:
    try:
        # Get last user message
        last_msg = [msg for speaker, msg in st.session_state.chat if speaker == "You"][-1]
        
        # Pass conversation history to process_user_input
        result = process_user_input(last_msg, st.session_state.conversation_history)
        
        if not isinstance(result, dict):
            raise ValueError("Invalid response format")
        
        ai_text = result.get("response", "I'm having trouble responding right now.")
        status = result.get("status", "Online")
        mode = result.get("mode", st.session_state.mode)
        
    except Exception as e:
        ai_text = f"⚠️ Internal error: {str(e)}"
        status = "Offline"
        mode = st.session_state.mode
    
    # Update conversation history
    st.session_state.conversation_history.append({"role": "user", "content": last_msg})
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_text})
    
    # Keep only last 20 messages (10 exchanges)
    if len(st.session_state.conversation_history) > 20:
        st.session_state.conversation_history = st.session_state.conversation_history[-20:]
    
    # Update session state
    st.session_state.status = status
    st.session_state.mode = mode
    st.session_state.chat.append(("AI", ai_text))
    st.session_state.thinking = False
    st.session_state.input_key += 1
    
    st.rerun()