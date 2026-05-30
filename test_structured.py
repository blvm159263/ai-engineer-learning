import os
import time
import anthropic
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Employee(BaseModel):
    name: str = Field(description="Full name of the employee")
    department: Optional[str] = Field(default=None, description="Department the employee belongs to")
    age: Optional[int] = Field(default=None, description="Age of the employee")
    major: Optional[str] = Field(default=None, description="Major field of study for the employee")
    
    @field_validator('age')
    @classmethod
    def age_must_be_realistic(cls, v):
        if v is not None and not (18 <= v <= 80):
            raise ValueError(f"Age {v} is not realistic for an employee")
        return v

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()
    
def parse_json_response(text: str) -> dict:
    # Strip markdown code block nếu có
    text = text.strip()
    if text.startswith("```"):
        # Xóa dòng đầu (```json hoặc ```) và dòng cuối (```)
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = re.sub(r'\n```$', '', text)
    return json.loads(text)

def chat_structured(user_input: str, retries: int = 3) -> Employee:
    last_error = None
    
    for attempt in range(retries):
        try:
            # Nếu có lỗi từ lần trước, đưa vào prompt để model tự fix
            error_context = ""
            if last_error:
                error_context = f"\n\nPrevious attempt failed validation: {last_error}\nPlease fix and return valid JSON."

            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                system="""Extract structured employee information from the user's input.
Return ONLY valid JSON. No explanation, no markdown, no extra text.""",
                messages=[{
                    "role": "user",
                    "content": f"""Extract employee information and return JSON:

Text: {user_input}

Required fields:
- name: string (full name, must not be null)
Optional fields:
- department: string
- age: integer (must be between 18-80)
- major: string
{error_context}"""
                }]
            )

            raw = response.content[0].text
            data = parse_json_response(raw)
            return Employee(**data)

        except ValidationError as e:
            last_error = str(e)
            print(f"Attempt {attempt + 1} failed validation: {last_error}")
            if attempt == retries - 1:
                raise Exception(f"Failed after {retries} attempts. Last error: {last_error}")

        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)

# Test the function
employee_info = chat_structured("he is a 11-year-old software engineer in the IT department. He studied Computer Science.")
print(employee_info)
print(f"Name: {employee_info.name}, Department: {employee_info.department}, Age: {employee_info.age}, Major: {employee_info.major}")