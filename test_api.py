import anthropic
import os
from dotenv import load_dotenv
import time

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat_stream(user_input: str):
    print(f"User: {user_input}")
    print("Assistant: ", end="", flush=True)
    
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system="You are a Frappe Framework expert.",
        messages=[{"role": "user", "content": user_input}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            time.sleep(0.05)
    
    print()  # newline sau khi stream xong

chat_stream("Explain Frappe's hook system in 3 sentences.")