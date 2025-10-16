import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Load your fine-tuned GPT-2 model
MODEL_NAME = "Omar-keita/gpt2-finetuned-maracademy"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Simple text cleaning
def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?.,!]', '', text)
    return text

# Generate bot response
def ask_bot(query, max_length=80):
    query = normalize_input(query)
    prompt = f"User: {query}\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
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
        response = response.split("Bot:")[-1].split("User:")[0].strip()
    if not response:
        response = "I'm sorry, I didn't understand that. Could you rephrase?"
    return response.capitalize()

# Streamlit app layout
st.title("🎓 MarAcademy Chatbot")
st.write("Ask me anything about Computer Science, MarAcademy, or mentorship. I'm here to help!")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = ask_bot(user_input)
        st.write(f"**Bot:** {response}")
    else:
        st.warning("Please enter a valid question.")
