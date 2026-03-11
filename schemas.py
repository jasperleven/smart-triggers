from pydantic import BaseModel

class CommentRequest(BaseModel):
    comment: str


class CommentResponse(BaseModel):
    trigger: str
    tone: str
    tone_percent: float
    avg_confidence: float