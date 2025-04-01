# Import necessary libraries for the script
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer  # Hugging Face tools for working with pre-trained models
import torch  # PyTorch library for tensor operations and model execution
import chainlit as cl  # Chainlit library for building conversational AI applications

# Define the model ID for the Hugging Face model to be used
model_id = "google/gemma-3-1b-it"  # Specify the pre-trained model from Hugging Face

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model for causal language modeling, specifying the device and data type
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map='cpu')

# Define a function to generate a reply based on a given prompt
def generate_reply(prompt):
    """
    Generates a text-based reply to the given prompt using a pre-trained language model.

    Args:
        prompt (str): The input text prompt to generate a reply for.

    Returns:
        str: The generated reply as a string, decoded from the model's output.

    Notes:
        - The function tokenizes the input prompt and processes it into tensors suitable for the model.
        - The model generates a response with specific parameters for sampling, including:
            - `max_new_tokens`: Limits the number of tokens in the generated output.
            - `do_sample`: Enables sampling for diverse outputs.
            - `temperature`: Controls randomness in generation (lower values make output more deterministic).
            - `top_p`: Implements nucleus sampling to limit the probability mass of considered tokens.
            - `top_k`: Limits the number of highest-probability tokens considered.
            - `repetition_penalty`: Penalizes repeated tokens to improve output diversity.
        - The output is decoded to a human-readable string, skipping special tokens.
    """
    # Tokenize the input prompt, preparing it for the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Move the tokenized inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Disable gradient computation for inference
    with torch.no_grad():
        # Generate a response using the model with specified parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # Limit the number of tokens in the generated output
            do_sample=True,  # Enable sampling for diverse outputs
            temperature=0.7,  # Control randomness in generation
            top_p=0.9,  # Use nucleus sampling to limit the probability mass of considered tokens
            top_k=50,  # Limit the number of highest-probability tokens considered
            repetition_penalty=1.1  # Penalize repeated tokens to improve diversity
        )
    # Decode the generated output into a human-readable string, skipping special tokens
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define an asynchronous function to handle incoming messages
@cl.on_message
async def handle_message(message: cl.Message):
    # Extract the content of the user's message
    user_prompt = message.content
    # Generate a reply using the `generate_reply` function
    reply = generate_reply(user_prompt)
    # Send the generated reply back to the user
    await cl.Message(content=reply).send()

# Define an asynchronous function to display model information when the chat starts
@cl.on_chat_start
async def show_model_info():
    # Create a message with the model's ID and type
    msg = f"🚀 Model loaded: `{model_id}`\n"
    msg += f"📦 Model type: `{model.config.model_type}`"
    # Send the message to the user
    await cl.Message(content=msg).send()

# Print the class name of the loaded model to the console
print(model.__class__.__name__)