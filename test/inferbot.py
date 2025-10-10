from transformers import AutoTokenizer, AutoModelForCausalLM
import os

base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = os.path.join(base_dir, 'gpt2-finetuned-maracademy')

# Load fine-tuned model and tokenizer
model_name = model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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

# Example usage
while True:
    user_question = input("You: ")
    if user_question.lower() in ["exit", "quit"]:
        break
    print("Bot:", ask_bot(user_question))