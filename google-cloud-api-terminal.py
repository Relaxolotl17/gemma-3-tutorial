import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyCXcq1cAwEs1JNSf6cSJx-vlXSgWWP2H6k")

# Initialize the GenerativeModel with the model name as a positional argument
model = genai.GenerativeModel("gemma-3-27b-it")

# Generate content
response = model.generate_content("What is the latest version of iphone?")

# Print the generated text
print(response.text)
