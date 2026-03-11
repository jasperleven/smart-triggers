from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import CommentRequest, CommentResponse
from ai import analyze_comment

app = FastAPI(title="Smart Triggers API")

# ===== CORS (обязательно для Tilda) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # позже сузим до домена
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Healthcheck =====
@app.get("/")
def root():
    return {"status": "ok"}

# ===== Analyze endpoint =====
@app.post("/analyze", response_model=CommentResponse)
async def analyze(request: CommentRequest):
    try:
        return analyze_comment(request.comment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Local run =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)