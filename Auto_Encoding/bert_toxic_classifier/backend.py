from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

clf = pipeline(
  'text-classification',
  "ZhengjunHUO/distilbert-toxicity-classifier", 
  use_fast=True
)

class Req(BaseModel):
    comment: str

@app.post("/classify")
def classify(request: Req):
    return clf(request.comment)