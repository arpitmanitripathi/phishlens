"""PhishLens web app.

    GET  /           -> the web page (app/static/index.html)
    GET  /health     -> {"status": "ok"}  (used by Docker / the host to check the app is alive)
    POST /api/check  -> {"url": "..."}  ->  verdict, risk score, reasons

Run locally:  uv run uvicorn app.main:app --reload    then open http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from phishlens.predict import Predictor

STATIC = Path(__file__).parent / "static"
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models ONCE when the server starts, not on every request.
    models["predictor"] = Predictor()
    yield
    models.clear()


app = FastAPI(
    title="PhishLens",
    description="Real-time phishing URL detection with an explainable LightGBM + CNN ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)


class CheckRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, examples=["http://paypa1-login.xyz"])


class ModelScores(BaseModel):
    lightgbm: float
    cnn: float


class CheckResponse(BaseModel):
    url: str  # the cleaned URL that was actually checked
    risk_score: int  # 0-100
    risk_level: str  # "Looks safe" | "Suspicious" | "Phishing"
    reasons: list[str]
    model_scores: ModelScores  # each model's own phishing probability (0-1)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/check", response_model=CheckResponse)
def check(request: CheckRequest):
    try:
        result = models["predictor"].score(request.url)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="This does not look like a website address (http or https).",
        ) from None
    return CheckResponse(
        url=result["url"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        reasons=result["reasons"],
        model_scores=ModelScores(lightgbm=result["lightgbm"], cnn=result["cnn"]),
    )
