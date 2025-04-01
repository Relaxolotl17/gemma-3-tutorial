import chainlit as cl
import google.generativeai as genai

# Configure the Generative AI client
genai.configure(api_key="[REPLACE-THIS-TEXT-WITH-YOUR-API-KEY]")

# Initialize the GenerativeModel
model = genai.GenerativeModel("gemma-3-27b-it")

@cl.on_message
async def handle_message(message: cl.Message):
    # Generate content based on the user's input
    response = model.generate_content(contents=[message.content])
    reply = response.text

    # Send the generated response back to the user
    await cl.Message(content=reply).send()
