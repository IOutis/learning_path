import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)

# Configure the Gemini API with the API key
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name= "gemini-2.0-flash-exp", system_instruction="""You are a highly detailed and thorough research assistant. When providing information:
- Always verify information across multiple sources
- Provide structured, comprehensive answers
- Include specific examples and code snippets when relevant
- Cross-reference all information
- Format output consistently using proper markdown
- Always cite sources inline as well as in references
""")


chat = model.start_chat(
    history=[
    ]
)

response = chat.send_message('What is the current price of tomato in telangana today is 13th Jan 2025. Search the web and answer the query accurately', stream=True)
for chunk in response:
    print(chunk.text, end="")

print(chat.history)