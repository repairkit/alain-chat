from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import chat
import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ログ設定
logger = logging.getLogger(__name__)

# Pydanticモデルの定義
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    content: str

#FastAPIのインスタンスを作成
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では具体的なドメインを指定
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE, OPTIONS等すべて許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received request: {request}")  # リクエスト全体をログ出力
        latest_message = request.messages[-1].content
        logger.info(f"Received message: {latest_message}")
        
        response = chat.main(latest_message)
        return ChatResponse(content=response)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail="メッセージの処理中にエラーが発生しました")

from typing import List




# サーバー起動部分
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)