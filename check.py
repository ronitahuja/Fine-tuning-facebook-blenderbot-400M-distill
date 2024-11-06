from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Path to the safetensors file
safetensors_path = './results'

# Load the model and tokenizer directly using safetensors
model = BlenderbotForConditionalGeneration.from_pretrained(safetensors_path)
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

# Function to generate a response
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    reply_ids = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,  # Enable sampling for more creative responses
        temperature=0.7,  # Control the randomness of the output
        top_p=0.9,  # Top-p sampling for better diversity
        top_k=50  # Restrict next word candidates for better control
    )
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

# Example loop to generate responses
user_input = input("User: ")
while True:
    response = generate_response(user_input)
    print("Bot:", response)

    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    user_input += f" User: {user_input} Bot: {response}"
