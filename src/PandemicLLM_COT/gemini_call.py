import google.generativeai as genai

# load in API key to call Gemini models
genai.configure(api_key="AIzaSyByh6W10hlX1KB_-AlvBDlAVJSFdZR6gHU")
model = genai.GenerativeModel('gemini-2.5-pro')

# Start a chat session
chat = model.start_chat(history=[])
# load the system prompt
from pathlib import Path
sys_prompt = Path('src/sys_prompt.txt').read_text()
print("Here is the completed system prompt:")
print(sys_prompt)
# pass the system prompt to Gemini
response = chat.send_message(sys_prompt)
print(f"User: Predict the trend of hospitalization by thinking step by step.")
print(f"Gemini: {response.text}")
print("-" * 20)