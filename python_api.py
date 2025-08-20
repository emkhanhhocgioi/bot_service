from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Giả sử bạn đã có hàm predict_intent từ intent_bot.py

from mongo_search_bot import search_trips_with_provinces , predict_intent

app = FastAPI(title="Intent Classification API")

class QueryIn(BaseModel):
    text: str
    use_transformer: Optional[bool] = True

class IntentOut(BaseModel):
    intent: str
    confidence: float


@app.get("/routes")
async def root(query: str):
    print(f"Received query: {query}")
    intent, confidence = predict_intent(query)
    print(f"Predicted intent: {intent} (confidence={confidence:.3f})")
    if intent == "ask_route" or intent == "ask_price" or intent == "ask_destination":
        result = search_trips_with_provinces(query, intent)
        return result
    ì

