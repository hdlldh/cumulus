from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

# load the model bert-base-uncased
model = pipeline('fill-mask', model='fill-mask-bert-base')

app = FastAPI()

class ModelInput(BaseModel):
    input: str

class ModelPrediction(BaseModel):
    score: float
    token: int
    token_str: str
    sequence: str

class ModelOutput(BaseModel):
    input: str
    output: List[ModelPrediction]

@app.get("/")
async def hello():
    return "use endpoint /unmask and provide a masked_str with [MASK] filling the token being masked."

@app.post("/unmask")
async def unmask(payload: ModelInput) -> ModelOutput:
    masked_str = payload.input
    preds = model(masked_str)
    preds.sort(key=lambda i: i['score'], reverse=True)
    unmasked_resp = [ModelPrediction.parse_obj(i) for i in preds]
    return ModelOutput(input=masked_str, output=unmasked_resp) 
