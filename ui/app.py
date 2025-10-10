from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


#================================================================
#file paths
#=================================================================
base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = os.path.join(base_dir, 'gpt2-finetuned-maracademy')

#==================================================================
#flask config
#==================================================================
app = Flask(__name__)




#===============================================================
# load model and ask the bot
#===============================================================
from transformers import AutoModel, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Omar-keita/maracademy-computer-science",
    subfolder="gpt2-finetuned-maracademy"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Omar-keita/maracademy-computer-science",
    subfolder="gpt2-finetuned-maracademy"
)
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






@app.route('/')
def home():
    return render_template('index.html')


#======================================================================
#Api
#======================================================================
@app.route('/api/chat', methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", '')
    if not user_message:
        return jsonify({"response": "Please send a message."})
    bot_response = ask_bot(user_message)
    return jsonify({"response": bot_response})
    




if __name__ == '__main__':
    app.run(debug=True)