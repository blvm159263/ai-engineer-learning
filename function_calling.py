import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Giả lập Frappe DB ---
FAKE_FRAPPE_DB = {
    "INV-2024-001": {"status": "overdue", "amount": 15000000, "currency": "VND", "customer": "ABC Corp", "days_overdue": 30},
    "INV-2024-002": {"status": "paid", "amount": 5000000, "currency": "VND", "customer": "XYZ Ltd", "days_overdue": 0},
    "INV-2024-003": {"status": "pending", "amount": 8500000, "currency": "VND", "customer": "DEF Inc", "days_overdue": 0},
}

# --- Tool definitions — model đọc cái này để biết có tool gì ---
tools = [
    {
        "name": "get_invoice",
        "description": "Get invoice details from Frappe ERP by invoice ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_id": {
                    "type": "string",
                    "description": "The invoice document name, e.g. INV-2024-001"
                }
            },
            "required": ["invoice_id"]
        }
    },
    {
        "name": "list_overdue_invoices",
        "description": "Get all overdue invoices from Frappe ERP",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# --- Actual functions bạn thực thi ---
def get_invoice(invoice_id: str) -> dict:
    invoice = FAKE_FRAPPE_DB.get(invoice_id)
    if not invoice:
        return {"error": f"Invoice {invoice_id} not found"}
    return {"invoice_id": invoice_id, **invoice}

def list_overdue_invoices() -> list:
    return [
        {"invoice_id": k, **v}
        for k, v in FAKE_FRAPPE_DB.items()
        if v["status"] == "overdue"
    ]

def run_tool(tool_name: str, tool_input: dict):
    if tool_name == "get_invoice":
        return get_invoice(**tool_input)
    elif tool_name == "list_overdue_invoices":
        return list_overdue_invoices()
    else:
        return {"error": f"Unknown tool: {tool_name}"}

# --- Agentic loop ---
def chat_with_tools(user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system="You are a Frappe ERP assistant. Use tools to answer questions about invoices.",
            tools=tools,
            messages=messages
        )

        print(f"\nStop reason: {response.stop_reason}")

        # Model trả lời xong, không cần tool
        if response.stop_reason == "end_turn":
            return response.content[0].text

        # Model muốn gọi tool
        if response.stop_reason == "tool_use":
            # Thêm response của model vào history
            messages.append({"role": "assistant", "content": response.content})

            # Xử lý từng tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Model calling tool: {block.name} with {block.input}")

                    # BẠN thực thi function
                    result = run_tool(block.name, block.input)
                    print(f"Tool result: {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            # Trả kết quả tool về cho model
            messages.append({"role": "user", "content": tool_results})
            # Loop lại — model sẽ dùng kết quả để trả lời


# Test
print("=== Test 1: Hỏi về invoice cụ thể ===")
print(chat_with_tools("What is the status of invoice INV-2024-001?"))

print("\n=== Test 2: Hỏi tổng quát ===")
print(chat_with_tools("Show me all overdue invoices"))

print("\n=== Test 3: Hỏi không có invoice ===")
print(chat_with_tools("Show me all employee older than 30."))
