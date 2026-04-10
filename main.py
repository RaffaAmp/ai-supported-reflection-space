# main.py
from typing import List, Literal
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Import shared config & helpers from Streamlit app
from app_test import (
    INSTRUCTIONS,
    MODEL,
    HISTORY_LENGTH,
    CONTEXT_LEN,
    load_lindenberg_knowledge_base,
    improved_search,
    build_prompt,
    history_to_text,
)

Role = Literal["user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
_last_request_time: datetime.datetime | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Framer domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_api_prompt(question: str, history: List[Message]) -> str:
    documents = load_lindenberg_knowledge_base()

    # limit history like in app_test.py
    recent_history = history[-HISTORY_LENGTH:] if history else []
    recent_history_str = history_to_text(
        [{"role": m.role, "content": m.content} for m in recent_history]
    ) if recent_history else None

    relevant_docs = improved_search(question, documents, max_results=CONTEXT_LEN)
    if relevant_docs:
        context_str = "\n\n".join(
            [f"Quelle: {doc['source']}\n{doc['content']}" for doc in relevant_docs]
        )
    else:
        context_str = "Keine relevanten Informationen in den Dokumenten gefunden."

    return build_prompt(
        instructions=INSTRUCTIONS,
        document_context=context_str,
        recent_messages=recent_history_str,
        question=question,
    )

def call_openai(prompt: str) -> str:
    client = OpenAI()  # uses OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1,
    )
    return resp.choices[0].message.content

@app.get("/")
def root():
    return {"status": "ok", "message": "Windpark Lindenberg backend is running."}

@app.post("/chat")
def chat(req: ChatRequest):
    global _last_request_time

    if not req.messages:
        return {"reply": "Bitte stellen Sie eine Frage zum Windpark Lindenberg."}

    # simple rate limit
    now = datetime.datetime.utcnow()
    if _last_request_time is not None:
        if now - _last_request_time < MIN_TIME_BETWEEN_REQUESTS:
            pass
    _last_request_time = now

    question = req.messages[-1].content
    history = req.messages[:-1]

    prompt = build_api_prompt(question, history)
    answer = call_openai(prompt)
    return {"reply": answer}
