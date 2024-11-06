from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

try:
    dataset = load_dataset('json', data_files='dataset.json')['train']
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']


def tokenize_function(examples):
    model_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply tokenization
try:
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
)

# Train model
try:
    trainer.train()

    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')
    print("Model saved successfully!")

except Exception as e:
    print(f"Error during training: {e}")
