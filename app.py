import streamlit as st
import json
from datetime import datetime
import sqlite3
import sys
import os

# --------------------------------------------------
# FIXED IMPORTS - Add current directory to path first
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import from main.py
try:
    from main import process_user_input
    from main import working_memory
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    IMPORT_SUCCESS = False
    
    # Create fallback functions if import fails
    def process_user_input(user_input, conversation_history=None):
        return {
            "response": "System initializing... Please run main.py directly.",
            "status": "Offline", 
            "mode": "personal"
        }
    
    working_memory = {"observations": []}

# Import Upstox auth
try:
    from financial.auth.upstox_auth import upstox_auth_flow, check_upstox_auth, UpstoxAuth
    UPSTOX_AVAILABLE = True
except ImportError:
    UPSTOX_AVAILABLE = False
    upstox_auth_flow = None
    check_upstox_auth = lambda: False
    UpstoxAuth = None

# --------------------------------------------------
# Page config (MUST be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="AI Personal Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show import status in sidebar for debugging
if not IMPORT_SUCCESS:
    st.sidebar.warning("⚠️ Could not import main module fully")

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

if "show_upstox_auth" not in st.session_state:
    st.session_state.show_upstox_auth = False

# --------------------------------------------------
# Handle Upstox authentication page
# --------------------------------------------------
if st.session_state.get('show_upstox_auth', False):
    if upstox_auth_flow:
        upstox_auth_flow()
    
    if st.button("← Back to Chat"):
        st.session_state.show_upstox_auth = False
        st.rerun()
    
    st.stop()  # Don't show chat interface while authenticating

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
            🔍 {st.session_state.mode.capitalize()} Mode
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # ========================================
    # UPSTOX AUTHENTICATION SECTION
    # ========================================
    st.markdown("### 🔐 Market Data Access")
    
    if UPSTOX_AVAILABLE:
        if check_upstox_auth():
            st.success("✅ Upstox Connected")
            
            if st.button("🚪 Logout Upstox", use_container_width=True):
                auth = UpstoxAuth()
                auth.logout()
                st.rerun()
        else:
            st.warning("⚠️ Not Authenticated")
            if st.button("🔐 Authenticate Upstox", type="primary", use_container_width=True):
                st.session_state.show_upstox_auth = True
                st.rerun()
    else:
        st.error("❌ Upstox SDK Missing")
        st.caption("Run: `pip install upstox-python-sdk`")
    
    st.markdown("---")
    
    # System info
    st.markdown("**🧠 Brain:** RouteLLM")
    st.markdown("**💾 Memory:** ON")
    st.markdown(f"**💬 Messages:** {len(st.session_state.chat) // 2}")
    
    st.markdown("---")
    
    # --------------------------------------------------
    # ENHANCED INTELLIGENCE SECTION - AS DROPDOWN
    # --------------------------------------------------
    with st.expander("🧠 Enhanced Intelligence", expanded=False):
        
        # Personality Insights
        if st.button("Show Personality Insights", use_container_width=True, key="personality_btn"):
            try:
                from main import ENHANCED_MEMORY_AVAILABLE, personality_engine
                if ENHANCED_MEMORY_AVAILABLE and personality_engine:
                    summary = personality_engine.get_personality_summary()
                    st.json(summary)
                else:
                    st.info("Personality engine not available.")
            except Exception as e:
                st.info(f"Personality engine not available: {str(e)[:50]}...")
        
        # Pattern Analysis
        if st.button("Show Behavior Patterns", use_container_width=True, key="patterns_btn"):
            try:
                from main import ENHANCED_MEMORY_AVAILABLE, pattern_recognizer
                if ENHANCED_MEMORY_AVAILABLE and pattern_recognizer:
                    patterns = pattern_recognizer.get_pattern_summary()
                    st.json(patterns)
                else:
                    st.info("Pattern recognizer not available.")
            except Exception as e:
                st.info(f"Pattern recognizer not available: {str(e)[:50]}...")
        
        # Goal Tracking
        if st.button("Manage Goals", use_container_width=True, key="goals_btn"):
            try:
                from main import ENHANCED_MEMORY_AVAILABLE, vector_memory
                if ENHANCED_MEMORY_AVAILABLE and vector_memory:
                    conn = sqlite3.connect(vector_memory.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM goals WHERE current_status = 'active'")
                    goals = cursor.fetchall()
                    conn.close()
                    
                    if goals:
                        for goal in goals:
                            progress = goal[5] if len(goal) > 5 else 0
                            st.write(f"**{goal[1]}**")
                            st.write(f"Description: {goal[2]}")
                            if goal[3]:
                                st.write(f"Target: {goal[3]}")
                            st.progress(progress)
                            st.write(f"Progress: {progress*100:.1f}%")
                            st.markdown("---")
                    else:
                        st.info("No active goals. Add one below!")
                    
                    # Add new goal form
                    with st.form("add_goal_form"):
                        goal_type = st.selectbox("Goal Type", ["Personal", "Health", "Career", "Learning", "Financial"])
                        description = st.text_input("Description")
                        target_date = st.date_input("Target Date (optional)")
                        
                        if st.form_submit_button("Add Goal"):
                            try:
                                vector_memory.add_goal(
                                    goal_type=goal_type,
                                    description=description,
                                    target_date=target_date.isoformat() if target_date else None
                                )
                                st.success("Goal added!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to add goal: {str(e)[:50]}...")
                else:
                    st.info("Goal tracking not available.")
            except Exception as e:
                st.info(f"Goal tracking not available: {str(e)[:50]}...")
        
        # Mood Tracking
        if st.button("Log Current Mood", use_container_width=True, key="mood_btn"):
            with st.form("mood_form"):
                mood_score = st.slider("Mood (1-10)", 1, 10, 5)
                energy_level = st.slider("Energy Level (1-10)", 1, 10, 5)
                emotion = st.selectbox("Dominant Emotion", 
                                     ["neutral", "happy", "sad", "angry", "excited", "calm", "anxious", "tired", "energetic"])
                context = st.text_input("Context (optional)", placeholder="e.g., 'after work', 'morning', 'before trading'")
                notes = st.text_area("Notes (optional)")
                
                if st.form_submit_button("Log Mood"):
                    try:
                        from main import ENHANCED_MEMORY_AVAILABLE, vector_memory
                        if ENHANCED_MEMORY_AVAILABLE and vector_memory:
                            vector_memory.log_mood(mood_score, energy_level, emotion, context, notes)
                            st.success("Mood logged successfully!")
                        else:
                            st.error("Vector memory not available.")
                    except Exception as e:
                        st.error(f"Failed to log mood: {str(e)[:50]}...")
    
    st.markdown("---")
    
    # Working memory status
    st.markdown("### 📄 Working Memory")
    try:
        obs_count = len(working_memory.get("observations", []))
        st.markdown(f"**Observations:** {obs_count}")
        
        if obs_count > 0:
            st.caption("Learning from your patterns...")
    except Exception:
        pass
    
    st.markdown("---")
    
    # File uploader in sidebar
    st.markdown("### 📎 Upload File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        key=f"file_{st.session_state.input_key}",
        help="Upload images, PDFs, documents, code files",
        label_visibility="collapsed"
    )
    
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
    # Chat input (stays fixed at bottom automatically)
    user_input = st.chat_input("Type your message here...")
    
# --------------------------------------------------
# Handle submission
# --------------------------------------------------
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
        ai_text = f"⚠️ Internal error: {str(e)[:100]}"
        status = "Offline"
        mode = st.session_state.mode
        print(f"Error in app.py processing: {e}")
    
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