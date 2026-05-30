import anthropic
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat_with_retry(user_input: str, max_retries: int = 3) -> str:
    messages = [{"role": "user", "content": user_input}]
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system="You are a Frappe Framework expert.",
                messages=messages
            )
            return response.content[0].text

        except anthropic.RateLimitError:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait)

        except anthropic.BadRequestError as e:
            # Không retry — request sai thì retry cũng vô ích
            print(f"Bad request: {e}")
            raise

        except anthropic.APIConnectionError:
            wait = 2 ** attempt
            print(f"Connection error. Waiting {wait}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                # Server error → retry
                wait = 2 ** attempt
                print(f"Server error {e.status_code}. Waiting {wait}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
            else:
                # Client error (4xx) → không retry
                print(f"Client error {e.status_code}: {e.message}")
                raise

    raise Exception(f"Failed after {max_retries} retries")


# Test happy path
result = chat_with_retry("What is Frappe's document lifecycle?")
print(result)