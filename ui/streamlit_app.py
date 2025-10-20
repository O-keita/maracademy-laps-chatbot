import streamlit as st
from streamlit_chat import message
from transformers import AutoTokenizer, AutoModelForCausalLM

#===================================================
# Page Configuration
#===================================================
st.set_page_config(page_title="MarAcademy Chatbot", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>MarAcademy Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about computer science or MarAcademy offerings, mentorship, scholarships, and more!</p>", unsafe_allow_html=True)

#===================================================
# Load Pretrained Model and Tokenizer
#===================================================
MODEL_NAME = "Omar-keita/gpt2-finetuned-maracademy"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

#===================================================
# Initialize Session State Variables
#===================================================
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

#===================================================
# Sidebar Configuration
#===================================================
st.sidebar.title("Sidebar")
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

#===================================================
# Bot Response Generation
#===================================================
def ask_bot(question, max_length=60):
    """
    Generate a response from the GPT-2 chatbot.
    
    Args:
    question (str): The user's question.
    max_length (int): Maximum length of the generated response.

    Returns:
    tuple: Response from the bot and the number of tokens used.
    """
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

#===================================================
# Response Container
#===================================================
response_container = st.container()

user_input = st.chat_input("Type your message and press Enter")
if user_input:
    bot_response, num_tokens = ask_bot(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(bot_response)
    cost = num_tokens * 0.002 / 1000  # Approximate cost per token
    st.session_state['cost'].append(cost)
    st.session_state['total_cost'] += cost

#===================================================
# Display Conversation
#===================================================
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
            message(st.session_state["generated"][i], key=f"{i}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")