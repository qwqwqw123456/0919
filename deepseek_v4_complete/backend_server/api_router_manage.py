from fastapi import APIRouter

router = APIRouter()
@router.get("/v1/chat")
def chat():
    return {"reply": "hello"}
