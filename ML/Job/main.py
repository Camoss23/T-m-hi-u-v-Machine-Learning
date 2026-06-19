from fastapi import FastAPI, HTTPException
from Schema import JobInput, PredictionOutput
import joblib
import pandas as pd
from contextlib import asynccontextmanager
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("model.pkl")
    print("Model loaded!")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Job classification",
    version="1.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: JobInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")

    df = pd.DataFrame([{
        "title": data.title,
        "description": data.description,
        "industry": data.industry,
        "location": data.location,
        "function": data.function
    }])

    prediction = model.predict(df)[0]
    return PredictionOutput(prediction=str(prediction))