import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with MarAcademy colors (Indigo/Dark Blue theme)
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
    
    [data-testid="stMain"] {
        background: transparent;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .header-section {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        animation: fadeIn 0.6s ease-in;
    }
    
    .header-section h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #4f46e5 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-section p {
        font-size: 1rem;
        color: #a0a0a0;
    }
    
    .messages-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 120px;
        animation: slideUp 0.5s ease-out;
    }
    
    .message-wrapper {
        display: flex;
        margin: 1rem 0;
        animation: messagePop 0.3s ease-out;
    }
    
    .message-wrapper.user {
        justify-content: flex-end;
    }
    
    .message-wrapper.bot {
        justify-content: flex-start;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 1.25rem;
        word-wrap: break-word;
    }
    
    .message-bubble.user {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border-top-right-radius: 0.25rem;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
    }
    
    .message-bubble.bot {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(79, 70, 229, 0.2);
        color: #e0e0e0;
        border-top-left-radius: 0.25rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .input-section {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(15, 23, 42, 0.98);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(79, 70, 229, 0.2);
        padding: 1.5rem;
        z-index: 100;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .input-container {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 1rem;
        align-items: flex-end;
    }
    
    .stTextInput {
        flex: 1;
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
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.4) !important;
    }
    
    .stTextInput input::placeholder {
        color: #808080 !important;
    }
    
    .stButton {
        flex-shrink: 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.3s ease !important;
        min-width: 80px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
        box-shadow: 0 0 30px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    .welcome-card {
        background: rgba(79, 70, 229, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 1.5rem;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        margin: 2rem 0;
    }
    
    .welcome-card h2 {
        color: #4f46e5;
        margin-bottom: 1rem;
    }
    
    .welcome-card p {
        color: #a0a0a0;
        line-height: 1.6;
    }
    
    .typing-indicator {
        display: flex;
        gap: 0.5rem;
        padding: 1rem 1.5rem;
    }
    
    .typing-dot {
        width: 0.5rem;
        height: 0.5rem;
        background: #808080;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes messagePop {
        0% {
            opacity: 0;
            transform: scale(0.8) translateY(10px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    @media (max-width: 768px) {
        .header-section h1 {
            font-size: 1.8rem;
        }
        
        .message-bubble {
            max-width: 85%;
            padding: 0.75rem 1rem;
        }
        
        .input-container {
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .stButton button {
            width: 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model():
    MODEL_NAME = "Umutoniwasepie/final_model"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?.,!]', '', text)
    return text

def capitalize_response(response):
    sentences = response.split(". ")
    unique_sentences = []
    for s in sentences:
        if s and s not in unique_sentences:
            unique_sentences.append(s.capitalize())
    return ". ".join(unique_sentences)

def test_query(query):
    query_lower = normalize_input(query)
    input_text = f"{query_lower}"
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).input_ids
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=200,
            temperature=0.8,
            top_k=70,
            repetition_penalty=1.5
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if not response.strip():
        response = "I'm sorry, I didn't understand that. Could you please rephrase?"
    
    return capitalize_response(response)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# Main layout
with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Header
    if st.session_state.show_welcome and len(st.session_state.conversation) == 0:
        st.markdown("""
            <div class="header-section">
                <h1>🤖 Customer Support</h1>
                <p>Your AI-powered customer support assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="welcome-card">
                <h2>Welcome! 👋</h2>
                <p>Ask me any customer support-related questions. I'm here to help! 
                <br><br><em style="color: #818cf8;">Note: I'm still learning, so I do have my limits.</em></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="header-section" style="padding: 1rem 0; margin-bottom: 1rem;">
                <h1 style="font-size: 2rem;">🤖 Customer Support</h1>
            </div>
        """, unsafe_allow_html=True)
    
    # Messages container
    st.markdown("<div class='messages-container'>", unsafe_allow_html=True)
    
    if len(st.session_state.conversation) > 0:
        st.session_state.show_welcome = False
        for i, (role, message) in enumerate(st.session_state.conversation):
            if role == "user":
                st.markdown(f"""
                    <div class="message-wrapper user">
                        <div class="message-bubble user">{message}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="message-wrapper bot">
                        <div class="message-bubble bot">{message}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Fixed input section at bottom
st.markdown("""
    <div class="input-section">
        <div class="input-container">
""", unsafe_allow_html=True)

col_input, col_button = st.columns([4, 1])

with col_input:
    user_input = st.text_input(
        "You:",
        placeholder="Ask your question here...",
        label_visibility="collapsed",
        key="user_input_field"
    )

with col_button:
    send_clicked = st.button("Send", use_container_width=True, key="send_button")

st.markdown("""
        </div>
    </div>
""", unsafe_allow_html=True)

# Add spacing to prevent overlap with fixed input
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Process message
if send_clicked and user_input.strip():
    # Add user message
    st.session_state.conversation.append(("user", user_input))
    
    # Show loading state
    with st.spinner(""):
        response = test_query(user_input)
    
    # Add bot response
    st.session_state.conversation.append(("bot", response))
    
    # Clear input and rerun
    st.rerun()

elif send_clicked and not user_input.strip():
    st.warning("Please enter a valid question.")