import streamlit as st
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set page title and header
st.set_page_config(page_title="MarAcademy Chatbot", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>MarAcademy Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about computer science or MarAcademy offerings, mentorship, scholarships, and more!</p>", unsafe_allow_html=True)

# Load model and tokenizer (from Hugging Face Hub or local path)
MODEL_NAME = "Omar-keita/gpt2-finetuned-maracademy"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - Clear chat option and info
st.sidebar.title("Sidebar")
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['total_tokens'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

# Bot response generation function
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
        return answer, len(output[0])
    return response, len(output[0])

# Streamlit UI - chat form
response_container = st.container()
container = st.container()

with container:
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        bot_response, num_tokens = ask_bot(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(bot_response)
        st.session_state['total_tokens'].append(num_tokens)
        # Estimate cost (example: assume $0.002 per 1K tokens, adjust as needed)
        cost = num_tokens * 0.002 / 1000
        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")