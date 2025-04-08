from fastapi import FastAPI

app = FastAPI()

@app.get("/add")
def add(x: int, y: int):
    return {"result": x + y}