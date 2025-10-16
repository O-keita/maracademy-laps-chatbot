import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MarAcademy Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern dark theme
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e0e0e0;
    }
    
    [data-testid="stChatMessageContainer"] {
        background: transparent;
    }
    
    .stChatMessage {
        background: transparent;
    }
    
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .header-section h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #4f46e5 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-section p {
        font-size: 1.1rem;
        color: #a0a0a0;
    }
    
    .input-section {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(79, 70, 229, 0.2);
        padding: 1.5rem;
        z-index: 100;
    }
    
    .message-user {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
    }
    
    .message-user .stChatMessage {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        border-radius: 1.25rem;
        border-top-right-radius: 0.25rem;
        padding: 1rem 1.5rem;
        max-width: 70%;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
    }
    
    .message-bot {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
    }
    
    .message-bot .stChatMessage {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 1.25rem;
        border-top-left-radius: 0.25rem;
        padding: 1rem 1.5rem;
        max-width: 70%;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stTextInput {
        margin-bottom: 2rem;
    }
    
    .stTextInput input {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(79, 70, 229, 0.3) !important;
        border-radius: 1rem !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border: 1px solid rgba(79, 70, 229, 0.6) !important;
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stTextInput input::placeholder {
        color: #808080 !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
        box-shadow: 0 0 30px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    .topic-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .topic-card:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(79, 70, 229, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2);
    }
    
    .topic-card h3 {
        color: #4f46e5;
        margin-bottom: 0.5rem;
    }
    
    .topic-card p {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .stat-box {
        background: rgba(79, 70, 229, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        flex: 1;
        min-width: 150px;
        backdrop-filter: blur(10px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }
    
    .stat-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    [data-testid="stChatMessageContent"] {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Load fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    MODEL_NAME = "Omar-keita/gpt2-finetuned-maracademy"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def ask_bot(question, max_length=60):
    prompt = f"User: {question}\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=30,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Bot:" in response:
        answer = response.split("Bot:")[1].split("User:")[0].strip()
        return answer
    return response

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# Main container
with st.container():
    # Welcome section
    if st.session_state.show_welcome and len(st.session_state.conversation) == 0:
        st.markdown("""
            <div class="header-section">
                <h1>🎓 Welcome to MarAcademy</h1>
                <p>Your personal computer science learning assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("""
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-number">1000+</div>
                    <div class="stat-label">Topics Covered</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Available</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">AI</div>
                    <div class="stat-label">Powered</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Quick Topics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="topic-card">
                    <h3>💻 Programming</h3>
                    <p>Learn coding concepts and best practices</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="topic-card">
                    <h3>🔍 Algorithms</h3>
                    <p>Master problem solving and optimization</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Chat history
    if len(st.session_state.conversation) > 0:
        st.session_state.show_welcome = False
    
    for role, message in st.session_state.conversation:
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
            st.markdown(message)

# Input section - Fixed at bottom
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your question:",
        placeholder="Ask me anything about CS, MarAcademy, mentorship...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Add padding to prevent content from being hidden behind fixed input
st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

# Process message
if send_button and user_input:
    st.session_state.conversation.append(("user", user_input))
    
    with st.spinner(""):
        bot_response = ask_bot(user_input)
    
    st.session_state.conversation.append(("assistant", bot_response))
    st.rerun()