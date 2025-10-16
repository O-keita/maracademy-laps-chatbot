import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="MarAcademy Chatbot 🎓", page_icon="🎓", layout="wide")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    MODEL_NAME = "Omar-keita/gpt2-finetuned-maracademy"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- Helper Function ----------------
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

# ---------------- Session State ----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ---------------- Custom Minimalist Styling ----------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    font-family: 'Inter', sans-serif;
    color: #1e1e1e;
}

h1 {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 2.5rem;
}

.chat-container {
    max-width: 750px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 1.5rem;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(79, 70, 229, 0.08);
}

.message {
    display: flex;
    margin: 1rem 0;
}

.message.user {
    justify-content: flex-end;
}

.message.bot {
    justify-content: flex-start;
}

.bubble {
    padding: 0.9rem 1.2rem;
    border-radius: 1.2rem;
    max-width: 70%;
    line-height: 1.5;
}

.user .bubble {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border-top-right-radius: 0.3rem;
}

.bot .bubble {
    background: #f1f5f9;
    border-top-left-radius: 0.3rem;
    color: #1e1e1e;
}

.input-container {
    margin-top: 2rem;
    text-align: center;
}

input[type="text"] {
    border-radius: 1rem !important;
    border: 1px solid #d1d5db !important;
    background-color: white !important;
    color: #111827 !important;
    padding: 0.7rem 1rem !important;
    font-size: 1rem !important;
    width: 80% !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important;
    color: white !important;
    border-radius: 1rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    margin-top: 0.5rem !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI Layout ----------------
st.markdown("<h1>MarAcademy Chatbot 🎓</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your personal Computer Science learning companion</p>", unsafe_allow_html=True)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display conversation
for role, msg in st.session_state.conversation:
    role_class = "user" if role == "user" else "bot"
    avatar = "👤" if role == "user" else "🤖"
    st.markdown(
        f"""
        <div class="message {role_class}">
            <div class="bubble">{avatar} {msg}</div>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Input Section ----------------
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
user_input = st.text_input("Ask me anything:", "", placeholder="E.g. Explain recursion, What is Python used for...", label_visibility="collapsed")
if st.button("Send"):
    if user_input.strip():
        st.session_state.conversation.append(("user", user_input))
        with st.spinner("Thinking..."):
            bot_response = ask_bot(user_input)
        st.session_state.conversation.append(("assistant", bot_response))
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
