import os
import json
import uuid
import httpx
import subprocess
from fastapi import FastAPI, Request
from openai import OpenAI
from google.cloud import storage

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # We can also use AIPipe proxy if needed
AIPIPE_URL = "https://aipipe.org/openai/v1"
PROJECT_ID = os.getenv("PROJECT_ID", "meridian-hackathon-2026")
BUCKET_NAME = os.getenv("BUCKET_NAME", "tds-telegram-logs-24f2001283")

# Initialize clients
client = OpenAI(
    base_url=AIPIPE_URL, 
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY
) if GEMINI_API_KEY else None

try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
except Exception:
    bucket = None

# In-memory storage for multi-turn chats
chat_histories = {}

def execute_python(code: str) -> str:
    with open("temp.py", "w") as f:
        f.write(code)
    try:
        res = subprocess.run(["python3", "temp.py"], capture_output=True, text=True, timeout=60)
        return res.stdout + res.stderr
    except Exception as e:
        return str(e)

def upload_log_to_gcs(log_data: list, run_id: str) -> str:
    if not bucket:
        return "https://example.com/log_not_available.jsonl"
    
    filename = f"runs/{run_id}.jsonl"
    blob = bucket.blob(filename)
    
    jsonl_str = "\n".join([json.dumps(m) for m in log_data])
    blob.upload_from_string(jsonl_str, content_type="application/jsonl")
    blob.make_public()
    return blob.public_url

def send_telegram_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        httpx.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending message: {e}")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    if "message" not in update:
        return {"status": "ok"}
    
    message = update["message"]
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    
    if not text:
        return {"status": "ok"}
        
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {"role": "system", "content": "You are a data-analyst agent. You have a python execution tool. Answer the user's question. You must work out the answer yourself (e.g. by writing python code to download and analyze data). At the very end, provide your final answer as a JSON object inside a ```json block with the key 'answer'."}
        ]
        
    chat_histories[chat_id].append({"role": "user", "content": text})
    
    messages = chat_histories[chat_id].copy()
    run_log = []
    run_log.extend(messages)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute python code and return stdout/stderr. Use it to download datasets and process data with pandas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    }]
    
    final_answer = None
    
    configs = [
        {"client": client, "model": "gpt-5-nano"},
        {"client": gemini_client, "model": "gemini-3.6-flash"},
        {"client": gemini_client, "model": "gemini-3.5-flash"},
        {"client": gemini_client, "model": "gemini-2.5-flash"},
        {"client": gemini_client, "model": "gemini-1.5-flash"}
    ]
    
    for config in configs:
        c = config["client"]
        m = config["model"]
        if not c:
            continue
            
        try:
            local_messages = messages.copy()
            local_run_log = run_log.copy()
            
            for _ in range(10): # max 10 steps
                response = c.chat.completions.create(
                    model=m,
                    messages=local_messages,
                    tools=tools
                )
                msg = response.choices[0].message
                local_messages.append(msg.model_dump(exclude_unset=True))
                local_run_log.append(msg.model_dump(exclude_unset=True))
                
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.function.name == "execute_python":
                            args = json.loads(tc.function.arguments)
                            out = execute_python(args["code"])
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.function.name,
                                "content": out
                            }
                            local_messages.append(tool_msg)
                            local_run_log.append(tool_msg)
                else:
                    final_answer = msg.content
                    break
                    
            if final_answer:
                messages = local_messages
                run_log = local_run_log
                break # successfully generated an answer
        except Exception as e:
            print(f"Model {m} failed: {e}")
            continue
            
    # Extract JSON from final_answer
    answer_obj = {}
    try:
        if "```json" in final_answer:
            json_str = final_answer.split("```json")[1].split("```")[0].strip()
            answer_obj = json.loads(json_str).get("answer", json.loads(json_str))
        else:
            answer_obj = json.loads(final_answer).get("answer", json.loads(final_answer))
    except Exception:
        answer_obj = {"raw": final_answer}
        
    run_id = uuid.uuid4().hex
    log_url = upload_log_to_gcs(run_log, run_id)
    
    final_reply = json.dumps({
        "answer": answer_obj,
        "log_url": log_url
    })
    
    chat_histories[chat_id].append({"role": "assistant", "content": final_reply})
    
    send_telegram_message(chat_id, final_reply)
    
    return {"status": "ok"}
