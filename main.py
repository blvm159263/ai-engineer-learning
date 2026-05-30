import anthropic
import os
from dotenv import load_dotenv
import time
from pydantic import BaseModel, Field, ValidationError, field_validator
import json
import re
from typing import Optional

from fastapi import FastAPI, HTTPException

app = FastAPI()

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Models ---
class InvoiceRequest(BaseModel):
    text: str

class InvoiceExtraction(BaseModel):
    invoice_id: Optional[str] = Field(None)
    amount: Optional[float] = Field(None)
    currency: Optional[str] = Field(None)
    status: Optional[str] = Field(None)
    customer: Optional[str] = Field(None)
    days_overdue: Optional[int] = Field(None)
    recommended_action: Optional[str] = Field(None)
    missing_info: Optional[str] = Field(None, description="Describe missing critical info or null")

class InvoiceAnalysis(BaseModel):
    invoice_id: str
    amount: float
    currency: str
    status: str
    customer: str
    days_overdue: Optional[int] = None
    recommended_action: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value):
        allowed = {"overdue", "paid", "pending", "cancelled"}
        if value not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return value

# --- Helpers ---
def parse_json_response(text: str) -> dict:
    # Strip markdown code block nếu có
    text = text.strip()
    if text.startswith("```"):
        # Xóa dòng đầu (```json hoặc ```) và dòng cuối (```)
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = re.sub(r'\n```$', '', text)
    return json.loads(text)

def call_llm(user_input: str, error_context: str = "") -> InvoiceExtraction:
    # Gọi API, trả về InvoiceExtraction
    # error_context được append vào prompt nếu có
    system_prompt = (
        "You are an expert at extracting structured data from unstructured text. "
        "Given a text describing an invoice, extract the following fields: "
        "invoice_id (string), amount (float), currency (string), status (overdue/paid/pending/cancelled), "
        "customer (string), days_overdue (int, optional), recommended_action (string, optional). "
        "If any critical information is missing, set that field to null and describe what's missing in 'missing_info'."
    )
    if error_context:
        system_prompt += f" Previous extraction attempt failed with error: {error_context}"
        
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}]
    )
    try:
        data = parse_json_response(response.content[0].text)
        return InvoiceExtraction(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise

def extract_with_retry(user_input: str, max_retries: int = 3) -> InvoiceExtraction:
    last_error = None
    for attempt in range(max_retries):
        try:
            result = call_llm(user_input, error_context=last_error or "")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            print(f"Attempt {attempt + 1} failed: {last_error}")
    raise Exception(f"Failed after {max_retries} attempts: {last_error}")

# --- Endpoint ---
@app.post("/extract-invoice")
def extract_invoice_endpoint(request: InvoiceRequest):
    try:
        extraction = extract_with_retry(request.text)
        
        if extraction.missing_info:
            raise HTTPException(status_code=422, detail=f"Insufficient information: {extraction.missing_info}")
        
        return InvoiceAnalysis(
            invoice_id=extraction.invoice_id,
            amount=extraction.amount,
            currency=extraction.currency,
            status=extraction.status,
            customer=extraction.customer,
            days_overdue=extraction.days_overdue,
            recommended_action=extraction.recommended_action
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))