from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import math
import os

# Define base directories
base_dir_data = os.path.join(os.path.dirname(__file__), '..', 'data')
base_dir_model = os.path.join(os.path.dirname(__file__), '..', 'models')
pretrain_corpus = os.path.join(base_dir_data, 'pretrain_corpus.txt')
gpt2_finetuned_maracademy = os.path.join(base_dir_model, 'gpt2-finetuned-maracademy')

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
train_file = pretrain_corpus

# Create training dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)

# Initialize data collator for batching
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=gpt2_finetuned_maracademy,
    num_train_epochs=20,
    per_device_train_batch_size=2,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# # Evaluate and calculate perplexity
# eval_results = trainer.evaluate()
# eval_loss = eval_results["eval_loss"]
# perplexity = math.exp(eval_loss)
# print(f"Perplexity: {perplexity:.2f}")

# Save the fine-tuned model and tokenizer
model.save_pretrained(gpt2_finetuned_maracademy)
tokenizer.save_pretrained(gpt2_finetuned_maracademy)
print("GPT-2 fine-tuned and saved.")