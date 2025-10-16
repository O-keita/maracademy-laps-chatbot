import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model and tokenizer from Hugging Face Hub
MODEL_NAME = "Omar-keita/gpt2-finetuned-maracademy"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def ask_bot(question, max_length=60):
    prompt = f"User: {question}\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1]+max_length,
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

st.set_page_config(page_title="MarAcademy Chatbot", page_icon="🤖")
st.title("MarAcademy Learning Analytics & Personalization Chatbot")
st.markdown("""
Ask questions about computer science or MarAcademy offerings, mentorship, scholarships, and more!
""")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("Your question:", "")

if st.button("Send") and user_input:
    bot_response = ask_bot(user_input)
    st.session_state.conversation.append(("User", user_input))
    st.session_state.conversation.append(("Bot", bot_response))

for who, msg in st.session_state.conversation:
    if who == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")