from transformers import pipeline
import torch
from PIL import Image

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cpu",
    torch_dtype=torch.bfloat16
)

image = Image.open("/home/proflead/Documents/Gemma-3/test.png")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is on the image?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
# Okay, let's take a look! 
# Based on the image, the animal on the candy is a **turtle**. 
# You can see the shell shape and the head and legs.
