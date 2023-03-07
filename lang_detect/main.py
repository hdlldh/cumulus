from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

# load the model bert-base-uncased
model = pipeline('text-classification', model='seq-cls-roberta-base')

app = FastAPI()

class ModelInput(BaseModel):
    input: str

class ModelPrediction(BaseModel):
    label: str
    score: float

class ModelOutput(BaseModel):
    input: str
    output: List[ModelPrediction]

@app.get("/")
async def hello():
    return "use endpoint /predict and provide a text."

@app.post("/predict")
async def predict(payload: ModelInput) -> ModelOutput:
    text = payload.input
    preds = model(text)
    preds.sort(key=lambda i: i['score'], reverse=True)
    resp = [ModelPrediction.parse_obj(i) for i in preds]
    return ModelOutput(input=text, output=resp) 
